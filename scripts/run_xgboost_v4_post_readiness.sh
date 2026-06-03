#!/bin/bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Run the post-readiness xgboost-v4 retrain and promotion-evidence pipeline.

The script refuses to start until the clean atomic live corpus reports
ready=true. By default it produces fresh Stage 1/2 evidence plus serving
readiness, then writes fail-closed promotion/objective audits that remain
blocked until shadow and cutover evidence are supplied.

Environment overrides:
  PYTHON_BIN              Python executable. Default: .venv/bin/python
  LIVE_ROOT               Clean 7-day live corpus root.
  STATUS_PATH             live-collection-status JSON path.
  MANIFEST_PATH           ETL processed-files manifest.
  LOG_DIR                 Live collector/scorer log directory.
  SCREEN_SESSION          Active screen session name.
  MONITORING_DB_PATH      MLOps DuckDB path.
  RUN_ID                  UTC run id. Default: current UTC timestamp.
  RUN_ROOT                Output root. Default: data/xgboost-v4-7d-run-${RUN_ID}
  POST_READINESS_STATE_DIR Directory for duplicate-run guard state.
  POST_READINESS_SENTINEL_PATH Completed-run sentinel path.
  POST_READINESS_LOCK_DIR  In-progress duplicate-run lock directory.
  POST_READINESS_LOCK_STALE_SECONDS Auto-clear abandoned lock after this age. Default: 86400
  POST_READINESS_LATEST_PATH Stable pointer to the latest completed run.
  FORCE_POST_READINESS_RERUN true to ignore the completed-run sentinel. Default: false
  CONTINUE_POST_READINESS_RUN true to reuse an existing RUN_ROOT for shadow/cutover. Default: false
  ENSEMBLE_SEEDS          xgboost-v4 ensemble seeds. Default: 0,17,42
  THRESHOLDS              Backtest threshold grid.
  FEE_BPS                 Backtest fee bps. Default: 10
  SLIPPAGE_BPS            Backtest slippage bps. Default: 5
  LATENCY_MS              Backtest latency ms. Default: 0
  RUN_SHADOW              true to run shadow-v1 and bootstrap. Default: false
  SHADOW_SINCE_MS         Required when RUN_SHADOW=true. Use auto with SHADOW_UNTIL_MS=auto
                          to derive the latest common feature window from STATUS_PATH.
  SHADOW_UNTIL_MS         Required when RUN_SHADOW=true. Use auto with SHADOW_SINCE_MS=auto
                          to derive the latest common feature window from STATUS_PATH.
  MIN_SHADOW_SESSION_SECONDS
                            Minimum shadow window duration. Default: 86400
  SHADOW_REPORT_PATH      Shadow report JSON path.
  SHADOW_EVALUATION_PATH  Shadow evaluation JSON path.
  BOOTSTRAP_DECISION_PATH Bootstrap decision JSON path.
  RUN_CUTOVER_REPORT      true to build Stage 5 cutover report. Default: false
  DRIFT_BASELINE_PATH     Cutover drift baseline JSON path.
  CUTOVER_REPORT_PATH     Stage 5 cutover report JSON path.
  SMOKE_PATH              Required cutover smoke JSON when RUN_CUTOVER_REPORT=true.
  GITHUB_ISSUE_CLOSURES_PATH Required Stage 5 GitHub issue closure evidence JSON when RUN_CUTOVER_REPORT=true.
  ISSUE_COVERAGE_AUDIT_PATH Issue-to-artifact audit JSON path.
  STRICT_FINAL_AUDIT      true makes final audits fail non-zero if blocked. Default: false
  PLAN_ONLY               true prints resolved paths/readiness and exits without writes.

Example:
  /bin/bash scripts/run_xgboost_v4_post_readiness.sh

With shadow evidence:
  RUN_SHADOW=true SHADOW_SINCE_MS=1779600000000 SHADOW_UNTIL_MS=1779686400000 \
    /bin/bash scripts/run_xgboost_v4_post_readiness.sh

With an auto-derived common feature window:
  RUN_SHADOW=true SHADOW_SINCE_MS=auto SHADOW_UNTIL_MS=auto \
    /bin/bash scripts/run_xgboost_v4_post_readiness.sh
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
LIVE_ROOT="${LIVE_ROOT:-data/live/xgboost-v4-multimarket-7d-atomic-20260523T125657Z}"
STATUS_PATH="${STATUS_PATH:-data/xgboost-v4-run-20260523T103814Z/artifacts/live_multimarket_7d_collection_status_latest.json}"
MANIFEST_PATH="${MANIFEST_PATH:-data/tmp/xgboost-v4-multimarket-7d-atomic-20260523T125657Z/etl-processed-files.txt}"
LOG_DIR="${LOG_DIR:-data/logs/champion-live-7d-atomic-20260523T125657Z}"
SCREEN_SESSION="${SCREEN_SESSION:-xgbv4_7d_atomic_20260523T125657Z}"
MONITORING_DB_PATH="${MONITORING_DB_PATH:-data/mlops/champion_catalog.duckdb}"
PROMOTION_PROCESS_PATH="${PROMOTION_PROCESS_PATH:-/Users/tcscoder/Downloads/champion-promotion.md}"
PROMOTION_REPO_RUNBOOK_PATH="${PROMOTION_REPO_RUNBOOK_PATH:-docs/runbooks/champion_promotion.md}"
MODEL_FAMILY="${MODEL_FAMILY:-btc-updown-15m}"
ENVIRONMENT="${ENVIRONMENT:-prod}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-data/xgboost-v4-7d-run-${RUN_ID}}"
POST_READINESS_STATE_DIR="${POST_READINESS_STATE_DIR:-data/tmp/xgboost-v4-multimarket-7d-atomic-20260523T125657Z}"
POST_READINESS_SENTINEL_PATH="${POST_READINESS_SENTINEL_PATH:-${POST_READINESS_STATE_DIR}/post-readiness-run-completed.json}"
POST_READINESS_LOCK_DIR="${POST_READINESS_LOCK_DIR:-${POST_READINESS_STATE_DIR}/post-readiness-run.lock}"
POST_READINESS_LOCK_STALE_SECONDS="${POST_READINESS_LOCK_STALE_SECONDS:-86400}"
POST_READINESS_LATEST_PATH="${POST_READINESS_LATEST_PATH:-data/xgboost-v4-run-20260523T103814Z/artifacts/xgboost_v4_post_readiness_latest.json}"
SLACK_DELIVERY_STATUS_PATH="${SLACK_DELIVERY_STATUS_PATH:-data/xgboost-v4-run-20260523T103814Z/artifacts/slack_status_delivery_latest.json}"
FORCE_POST_READINESS_RERUN="${FORCE_POST_READINESS_RERUN:-false}"
CONTINUE_POST_READINESS_RUN="${CONTINUE_POST_READINESS_RUN:-false}"
ENSEMBLE_SEEDS="${ENSEMBLE_SEEDS:-0,17,42}"
LABEL_LOOKBACK_MINUTES="${LABEL_LOOKBACK_MINUTES:-10200}"
LABEL_REQUEST_TIMEOUT_SECONDS="${LABEL_REQUEST_TIMEOUT_SECONDS:-12}"
LABEL_REQUEST_CONCURRENCY="${LABEL_REQUEST_CONCURRENCY:-4}"
THRESHOLDS="${THRESHOLDS:-0.00,0.03,0.05,0.10,0.20,0.30,0.45}"
FEE_BPS="${FEE_BPS:-10}"
SLIPPAGE_BPS="${SLIPPAGE_BPS:-5}"
LATENCY_MS="${LATENCY_MS:-0}"
SERVING_SAMPLE_SIZE="${SERVING_SAMPLE_SIZE:-1000}"
SERVING_BATCH_SIZES="${SERVING_BATCH_SIZES:-10000,100000}"
RUN_SHADOW="${RUN_SHADOW:-false}"
SHADOW_SINCE_MS="${SHADOW_SINCE_MS:-}"
SHADOW_UNTIL_MS="${SHADOW_UNTIL_MS:-}"
SHADOW_EDGE_THRESHOLD="${SHADOW_EDGE_THRESHOLD:-0.30}"
MIN_SHADOW_SESSION_SECONDS="${MIN_SHADOW_SESSION_SECONDS:-86400}"
RUN_CUTOVER_REPORT="${RUN_CUTOVER_REPORT:-false}"
STRICT_FINAL_AUDIT="${STRICT_FINAL_AUDIT:-false}"
PLAN_ONLY="${PLAN_ONLY:-false}"

ARTIFACT_ROOT="${RUN_ROOT}/artifacts"
TRAINING_DIR="${ARTIFACT_ROOT}/training"
TRAINING_DOWN_DIR="${ARTIFACT_ROOT}/training-down-validation"
MODEL_ROOT="${ARTIFACT_ROOT}/models"
STABILITY_ROOT="${ARTIFACT_ROOT}/dataset-stability"
BACKTEST_ROOT="${ARTIFACT_ROOT}/backtests"
SHADOW_ROOT="${ARTIFACT_ROOT}/shadow"
BOOTSTRAP_ROOT="${ARTIFACT_ROOT}/bootstrap"
CUTOVER_ROOT="${ARTIFACT_ROOT}/cutover"
AUDIT_ROOT="${ARTIFACT_ROOT}/champion-promotion-audit"
OBJECTIVE_AUDIT_PATH="${ARTIFACT_ROOT}/xgboost_v4_objective_audit.json"
ISSUE_COVERAGE_AUDIT_PATH="${ISSUE_COVERAGE_AUDIT_PATH:-${ARTIFACT_ROOT}/issue_coverage_audit.json}"
SNAPSHOT_PATH="${ARTIFACT_ROOT}/current_champion_before_retrain.json"
RUN_MANIFEST_PATH="${ARTIFACT_ROOT}/run_manifest.json"
SHADOW_REPORT_PATH="${SHADOW_REPORT_PATH:-${SHADOW_ROOT}/xgboost-v4-shadow-report.json}"
SHADOW_EVALUATION_PATH="${SHADOW_EVALUATION_PATH:-${SHADOW_ROOT}/xgboost-v4-shadow-evaluation.json}"
BOOTSTRAP_DECISION_PATH="${BOOTSTRAP_DECISION_PATH:-${BOOTSTRAP_ROOT}/bootstrap_decision.json}"
DRIFT_BASELINE_PATH="${DRIFT_BASELINE_PATH:-${CUTOVER_ROOT}/drift-baseline-xgboost-v4.json}"
CUTOVER_REPORT_PATH="${CUTOVER_REPORT_PATH:-${CUTOVER_ROOT}/xgboost-v4-cutover.json}"
SMOKE_PATH="${SMOKE_PATH:-${CUTOVER_ROOT}/inference-smoke.json}"
GITHUB_ISSUE_CLOSURES_PATH="${GITHUB_ISSUE_CLOSURES_PATH:-${CUTOVER_ROOT}/github-issue-closures.json}"
POST_READINESS_LOCK_ACQUIRED="false"

export PYTHONPATH="${PYTHONPATH:-src}"

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "${path}" ]]; then
    echo "[xgbv4-post] missing ${label}: ${path}" >&2
    exit 1
  fi
}

require_command() {
  local command="$1"
  if ! command -v "${command}" >/dev/null 2>&1; then
    echo "[xgbv4-post] missing command: ${command}" >&2
    exit 1
  fi
}

run() {
  echo
  echo "[xgbv4-post] $*"
  "$@"
}

run_quiet() {
  echo
  echo "[xgbv4-post] $*"
  "$@" >/dev/null
}

json_get() {
  local path="$1"
  local dotted_key="$2"
  "${PYTHON_BIN}" - "${path}" "${dotted_key}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
value = payload
for key in sys.argv[2].split("."):
    if not isinstance(value, dict):
        value = None
        break
    value = value.get(key)
if value is None:
    print("")
elif isinstance(value, bool):
    print("true" if value else "false")
else:
    print(value)
PY
}

readiness_is_true() {
  "${PYTHON_BIN}" -c '
import json
import sys

payload = json.load(sys.stdin)
if payload.get("ready") is True or payload.get("ready_for_training") is True:
    raise SystemExit(0)
print(json.dumps(payload, indent=2, sort_keys=True))
raise SystemExit(1)
'
}

require_live_root_configuration() {
  local live_root_name
  live_root_name="$(basename "${LIVE_ROOT}")"
  if [[ "${live_root_name}" != xgboost-v4-*multimarket-7d-atomic-* ]]; then
    echo "[xgbv4-post] LIVE_ROOT must point at the clean xgboost-v4 multimarket 7d atomic corpus root, got: ${LIVE_ROOT}" >&2
    exit 1
  fi
  if [[ -z "${SCREEN_SESSION}" ]]; then
    echo "[xgbv4-post] SCREEN_SESSION must be set" >&2
    exit 1
  fi
}

require_status_matches_live_root() {
  "${PYTHON_BIN}" - "${STATUS_PATH}" "${LIVE_ROOT}" "${SCREEN_SESSION}" <<'PY'
import json
import sys
from pathlib import Path

status_path = Path(sys.argv[1])
live_root = sys.argv[2].rstrip("/")
screen_session = sys.argv[3]
expected_warehouse = f"{live_root}/warehouse"

try:
    payload = json.loads(status_path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    print(f"[xgbv4-post] unable to read live status JSON {status_path}: {exc}", file=sys.stderr)
    raise SystemExit(1)

if not isinstance(payload, dict):
    print(f"[xgbv4-post] live status JSON must be an object: {status_path}", file=sys.stderr)
    raise SystemExit(1)

issues = []
status_live_root = str(payload.get("live_root", "")).rstrip("/")
if status_live_root != live_root:
    issues.append(f"status live_root {status_live_root!r} does not match LIVE_ROOT {live_root!r}")

status_warehouse = str(payload.get("warehouse", "")).rstrip("/")
if status_warehouse != expected_warehouse:
    issues.append(
        f"status warehouse {status_warehouse!r} does not match expected {expected_warehouse!r}"
    )

status_screen = payload.get("screen_session")
if status_screen != screen_session:
    issues.append(
        f"status screen_session {status_screen!r} does not match SCREEN_SESSION {screen_session!r}"
    )

if issues:
    print(
        "[xgbv4-post] live status artifact does not describe the configured clean corpus:",
        file=sys.stderr,
    )
    for issue in issues:
        print(f"  - {issue}", file=sys.stderr)
    raise SystemExit(1)
PY
}

check_status_disk_headroom() {
  "${PYTHON_BIN}" - "${STATUS_PATH}" <<'PY'
import json
import os
import sys
from pathlib import Path

status_path = Path(sys.argv[1])
try:
    payload = json.loads(status_path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    print(f"[xgbv4-post] unable to read live status JSON {status_path}: {exc}", file=sys.stderr)
    raise SystemExit(1)

if not isinstance(payload, dict):
    print(f"[xgbv4-post] live status JSON must be an object: {status_path}", file=sys.stderr)
    raise SystemExit(1)

disk = payload.get("disk_headroom_evidence")
if not isinstance(disk, dict):
    raise SystemExit(0)


def bool_value(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return None


def gib(value: object) -> str:
    try:
        return f"{float(value) / 1024**3:.2f}GiB"
    except (TypeError, ValueError):
        return "n/a"


headroom_ok = bool_value(disk.get("headroom_ok"))
headroom_low_margin = bool_value(disk.get("headroom_low_margin"))
free_bytes = disk.get("free_bytes")
required_bytes = disk.get("required_free_bytes")
margin_bytes = disk.get("headroom_margin_bytes")
low_margin_threshold_bytes = disk.get("low_margin_threshold_bytes")

if headroom_ok is False:
    print(
        "[xgbv4-post] disk headroom is blocked; refusing post-readiness run "
        f"(free={gib(free_bytes)}, required={gib(required_bytes)}, margin={gib(margin_bytes)})",
        file=sys.stderr,
    )
    raise SystemExit(2)

if headroom_low_margin is True:
    print(
        "[xgbv4-post] WARNING: disk headroom passes but margin is low "
        f"(free={gib(free_bytes)}, required={gib(required_bytes)}, margin={gib(margin_bytes)})"
    )

try:
    required_int = int(required_bytes)
except (TypeError, ValueError):
    required_int = None

try:
    low_margin_threshold_int = int(low_margin_threshold_bytes)
except (TypeError, ValueError):
    low_margin_threshold_int = None

current_path = Path(str(disk.get("path") or payload.get("live_root") or "."))
if required_int is not None and current_path.exists():
    stat = os.statvfs(current_path)
    current_free_bytes = stat.f_bavail * stat.f_frsize
    current_margin_bytes = current_free_bytes - required_int
    if current_margin_bytes < 0:
        print(
            "[xgbv4-post] current filesystem disk headroom is blocked; "
            "refusing post-readiness run "
            f"(path={current_path}, free={gib(current_free_bytes)}, "
            f"required={gib(required_int)}, margin={gib(current_margin_bytes)})",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if (
        low_margin_threshold_int is not None
        and current_margin_bytes < low_margin_threshold_int
    ):
        print(
            "[xgbv4-post] WARNING: current filesystem disk headroom passes "
            "but margin is low "
            f"(path={current_path}, free={gib(current_free_bytes)}, "
            f"required={gib(required_int)}, margin={gib(current_margin_bytes)})"
        )
PY
}

require_bool_flag() {
  local name="$1"
  local value="$2"
  if [[ "${value}" != "true" && "${value}" != "false" ]]; then
    echo "[xgbv4-post] ${name} must be true or false, got: ${value}" >&2
    exit 1
  fi
}

validate_mode_flags() {
  require_bool_flag "PLAN_ONLY" "${PLAN_ONLY}"
  require_bool_flag "RUN_SHADOW" "${RUN_SHADOW}"
  require_bool_flag "RUN_CUTOVER_REPORT" "${RUN_CUTOVER_REPORT}"
  require_bool_flag "STRICT_FINAL_AUDIT" "${STRICT_FINAL_AUDIT}"
  require_bool_flag "FORCE_POST_READINESS_RERUN" "${FORCE_POST_READINESS_RERUN}"
  require_bool_flag "CONTINUE_POST_READINESS_RUN" "${CONTINUE_POST_READINESS_RUN}"
  if ! [[ "${POST_READINESS_LOCK_STALE_SECONDS}" =~ ^[0-9]+$ ]]; then
    echo "[xgbv4-post] POST_READINESS_LOCK_STALE_SECONDS must be a non-negative integer, got: ${POST_READINESS_LOCK_STALE_SECONDS}" >&2
    exit 1
  fi
  if ! [[ "${MIN_SHADOW_SESSION_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[xgbv4-post] MIN_SHADOW_SESSION_SECONDS must be a positive integer, got: ${MIN_SHADOW_SESSION_SECONDS}" >&2
    exit 1
  fi

  if [[ "${CONTINUE_POST_READINESS_RUN}" == "true" && "${RUN_SHADOW}" != "true" && "${RUN_CUTOVER_REPORT}" != "true" ]]; then
    echo "[xgbv4-post] CONTINUE_POST_READINESS_RUN=true requires RUN_SHADOW=true or RUN_CUTOVER_REPORT=true" >&2
    exit 1
  fi

  if [[ "${RUN_SHADOW}" == "true" && ( -z "${SHADOW_SINCE_MS}" || -z "${SHADOW_UNTIL_MS}" ) ]]; then
    echo "[xgbv4-post] RUN_SHADOW=true requires SHADOW_SINCE_MS and SHADOW_UNTIL_MS" >&2
    exit 1
  fi
  if [[ "${RUN_SHADOW}" == "true" ]]; then
    if [[ "${SHADOW_SINCE_MS}" == "auto" || "${SHADOW_UNTIL_MS}" == "auto" ]]; then
      if [[ "${SHADOW_SINCE_MS}" != "auto" || "${SHADOW_UNTIL_MS}" != "auto" ]]; then
        echo "[xgbv4-post] use auto for both SHADOW_SINCE_MS and SHADOW_UNTIL_MS, or provide both as integer epoch milliseconds" >&2
        exit 1
      fi
    else
      if ! [[ "${SHADOW_SINCE_MS}" =~ ^[0-9]+$ && "${SHADOW_UNTIL_MS}" =~ ^[0-9]+$ ]]; then
        echo "[xgbv4-post] SHADOW_SINCE_MS and SHADOW_UNTIL_MS must be integer epoch milliseconds" >&2
        exit 1
      fi
      local shadow_duration_ms
      local min_shadow_duration_ms
      shadow_duration_ms=$((SHADOW_UNTIL_MS - SHADOW_SINCE_MS))
      min_shadow_duration_ms=$((MIN_SHADOW_SESSION_SECONDS * 1000))
      if (( shadow_duration_ms < min_shadow_duration_ms )); then
        echo "[xgbv4-post] shadow window is too short: duration_ms=${shadow_duration_ms}, required_ms>=${min_shadow_duration_ms}" >&2
        exit 1
      fi
    fi
  fi

  if [[ "${RUN_SHADOW}" == "true" && "${BOOTSTRAP_DECISION_PATH}" != "${BOOTSTRAP_ROOT}/bootstrap_decision.json" ]]; then
    echo "[xgbv4-post] RUN_SHADOW=true writes bootstrap output to ${BOOTSTRAP_ROOT}/bootstrap_decision.json; BOOTSTRAP_DECISION_PATH must match or RUN_SHADOW=false must use an existing artifact" >&2
    exit 1
  fi

  if [[ "${RUN_CUTOVER_REPORT}" == "true" ]]; then
    require_file "${SMOKE_PATH}" "cutover smoke artifact"
    require_file "${GITHUB_ISSUE_CLOSURES_PATH}" "GitHub issue closure evidence"
    if [[ "${RUN_SHADOW}" != "true" ]]; then
      require_file "${BOOTSTRAP_DECISION_PATH}" "bootstrap decision"
      require_file "${SHADOW_EVALUATION_PATH}" "shadow evaluation"
    fi
  fi
}

resolve_shadow_window_if_auto() {
  if [[ "${RUN_SHADOW}" != "true" || "${SHADOW_SINCE_MS}" != "auto" || "${SHADOW_UNTIL_MS}" != "auto" ]]; then
    return
  fi
  require_file "${STATUS_PATH}" "live status for auto shadow window"
  local resolved
  resolved="$(
    "${PYTHON_BIN}" - "${STATUS_PATH}" "${MIN_SHADOW_SESSION_SECONDS}" <<'PY'
import json
import sys
from pathlib import Path

status_path = Path(sys.argv[1])
min_shadow_session_seconds = int(sys.argv[2])
payload = json.loads(status_path.read_text(encoding="utf-8"))
readiness = payload.get("collection_readiness") if isinstance(payload.get("collection_readiness"), dict) else {}
spans = payload.get("family_spans") if isinstance(payload.get("family_spans"), dict) else {}
feature_spans = spans.get("features_15m_v1") if isinstance(spans.get("features_15m_v1"), dict) else {}
required_families = readiness.get("required_families")
if not isinstance(required_families, list) or not required_families:
    required_families = sorted(feature_spans)

issues: list[str] = []
starts: list[int] = []
ends: list[int] = []
for family in required_families:
    span = feature_spans.get(str(family))
    if not isinstance(span, dict):
        issues.append(f"{family}: missing feature span")
        continue
    try:
        start = int(span["min_ts"])
        end = int(span["max_ts"])
    except (KeyError, TypeError, ValueError):
        issues.append(f"{family}: invalid min_ts/max_ts")
        continue
    starts.append(start)
    ends.append(end)

if issues:
    print("[xgbv4-post] cannot resolve auto shadow window:", file=sys.stderr)
    for issue in issues:
        print(f"  - {issue}", file=sys.stderr)
    raise SystemExit(1)
if not starts or not ends:
    print("[xgbv4-post] cannot resolve auto shadow window: no feature spans", file=sys.stderr)
    raise SystemExit(1)

common_start = max(starts)
common_end = min(ends)
required_ms = min_shadow_session_seconds * 1000
available_ms = common_end - common_start
if available_ms < required_ms:
    print(
        "[xgbv4-post] cannot resolve auto shadow window: "
        f"common feature span {available_ms}ms below required {required_ms}ms",
        file=sys.stderr,
    )
    raise SystemExit(1)

since_ms = common_end - required_ms
print(f"{since_ms} {common_end}")
PY
  )"
  SHADOW_SINCE_MS="${resolved%% *}"
  SHADOW_UNTIL_MS="${resolved##* }"
  echo "[xgbv4-post] auto shadow window resolved: since_ms=${SHADOW_SINCE_MS} until_ms=${SHADOW_UNTIL_MS} duration_s=${MIN_SHADOW_SESSION_SECONDS}"
}

cleanup_post_readiness_lock() {
  if [[ "${POST_READINESS_LOCK_ACQUIRED}" == "true" ]]; then
    rmdir "${POST_READINESS_LOCK_DIR}" 2>/dev/null || true
    rm -f "${POST_READINESS_LOCK_DIR}.json" 2>/dev/null || true
  fi
}

trap cleanup_post_readiness_lock EXIT

clear_stale_post_readiness_lock() {
  "${PYTHON_BIN}" - "${POST_READINESS_LOCK_DIR}" "${POST_READINESS_LOCK_DIR}.json" "${POST_READINESS_LOCK_STALE_SECONDS}" <<'PY'
import sys
import time
from pathlib import Path

lock_dir = Path(sys.argv[1])
metadata_path = Path(sys.argv[2])
stale_seconds = int(sys.argv[3])
if stale_seconds <= 0 or not lock_dir.exists():
    raise SystemExit(0)

paths = [lock_dir]
if metadata_path.exists():
    paths.append(metadata_path)
latest_mtime = max(path.stat().st_mtime for path in paths)
age_seconds = time.time() - latest_mtime
if age_seconds < stale_seconds:
    raise SystemExit(0)

try:
    lock_dir.rmdir()
except OSError:
    raise SystemExit(0)

try:
    metadata_path.unlink()
except FileNotFoundError:
    pass

print(
    f"[xgbv4-post] removed stale post-readiness lock: {lock_dir} "
    f"age_seconds={age_seconds:.0f}"
)
PY
}

claim_post_readiness_run() {
  if [[ "${FORCE_POST_READINESS_RERUN}" != "true" && "${CONTINUE_POST_READINESS_RUN}" != "true" && -f "${POST_READINESS_SENTINEL_PATH}" ]]; then
    if [[ "${RUN_SHADOW}" == "true" || "${RUN_CUTOVER_REPORT}" == "true" ]]; then
      echo "[xgbv4-post] post-readiness run already completed; refusing to silently skip requested shadow/cutover work. Use CONTINUE_POST_READINESS_RUN=true with the existing RUN_ROOT, FORCE_POST_READINESS_RERUN=true to regenerate the full pipeline, or choose a fresh RUN_ROOT." >&2
      return 2
    fi
    echo "[xgbv4-post] post-readiness run already completed; sentinel=${POST_READINESS_SENTINEL_PATH}"
    return 1
  fi

  mkdir -p "$(dirname "${POST_READINESS_LOCK_DIR}")"
  if [[ "${FORCE_POST_READINESS_RERUN}" == "true" ]]; then
    rmdir "${POST_READINESS_LOCK_DIR}" 2>/dev/null || true
  else
    clear_stale_post_readiness_lock
  fi
  if ! mkdir "${POST_READINESS_LOCK_DIR}" 2>/dev/null; then
    echo "[xgbv4-post] post-readiness run already in progress; lock=${POST_READINESS_LOCK_DIR}"
    return 1
  fi
  POST_READINESS_LOCK_ACQUIRED="true"

  "${PYTHON_BIN}" - "${POST_READINESS_LOCK_DIR}.json" \
    "run_id=${RUN_ID}" \
    "run_root=${RUN_ROOT}" \
    "sentinel_path=${POST_READINESS_SENTINEL_PATH}" <<'PY'
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

path = Path(sys.argv[1])
values = dict(arg.split("=", 1) for arg in sys.argv[2:])


def write_json_atomic(target: Path, payload: dict) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(target)
    finally:
        if tmp.exists():
            tmp.unlink()


payload = {
    "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    "run_id": values["run_id"],
    "run_root": values["run_root"],
    "sentinel_path": values["sentinel_path"],
}
write_json_atomic(path, payload)
PY
  return 0
}

print_plan() {
  local readiness_json="$1"

  echo
  echo "[xgbv4-post] PLAN_ONLY=true; no labels, training, backtests, or audits will run"
  echo "[xgbv4-post] current readiness:"
  printf '%s\n' "${readiness_json}"
  echo
  echo "[xgbv4-post] resolved artifact paths:"
  echo "  status=${STATUS_PATH}"
  echo "  run_root=${RUN_ROOT}"
  echo "  run_manifest=${RUN_MANIFEST_PATH}"
  echo "  post_readiness_sentinel=${POST_READINESS_SENTINEL_PATH}"
  echo "  post_readiness_lock=${POST_READINESS_LOCK_DIR}"
  echo "  post_readiness_latest=${POST_READINESS_LATEST_PATH}"
  echo "  slack_delivery_status=${SLACK_DELIVERY_STATUS_PATH}"
  echo "  promotion_process=${PROMOTION_PROCESS_PATH}"
  echo "  promotion_repo_runbook=${PROMOTION_REPO_RUNBOOK_PATH}"
  echo "  snapshot=${SNAPSHOT_PATH}"
  echo "  training=${TRAINING_DIR}"
  echo "  training_down_validation=${TRAINING_DOWN_DIR}"
  echo "  incumbent_eval=${MODEL_ROOT}/incumbent-same-dataset"
  echo "  candidate_model=${MODEL_ROOT}/xgboost-v4/model.json"
  echo "  candidate_calibration=${MODEL_ROOT}/xgboost-v4-calibration/calibration.json"
  echo "  candidate_eval=${MODEL_ROOT}/xgboost-v4-same-dataset"
  echo "  stability_report=${STABILITY_ROOT}/dataset_stability_report.json"
  echo "  rerun_report=${ARTIFACT_ROOT}/rerun_report.md"
  echo "  feature_ablation=${MODEL_ROOT}/xgboost-v4-feature-ablation/feature_ablation.json"
  echo "  incumbent_backtest=${BACKTEST_ROOT}/incumbent/summary.json"
  echo "  candidate_backtest=${BACKTEST_ROOT}/xgboost-v4/summary.json"
  echo "  down_validation=${BACKTEST_ROOT}/xgboost-v4-down/diagnostics.json"
  echo "  serving_readiness=${MODEL_ROOT}/xgboost-v4-serving-readiness.json"
  echo "  shadow_report=${SHADOW_REPORT_PATH}"
  echo "  shadow_evaluation=${SHADOW_EVALUATION_PATH}"
  echo "  bootstrap_decision=${BOOTSTRAP_DECISION_PATH}"
  echo "  drift_baseline=${DRIFT_BASELINE_PATH}"
  echo "  cutover_report=${CUTOVER_REPORT_PATH}"
  echo "  github_issue_closures=${GITHUB_ISSUE_CLOSURES_PATH}"
  echo "  promotion_audit=${AUDIT_ROOT}/champion_promotion_audit.json"
  echo "  objective_audit=${OBJECTIVE_AUDIT_PATH}"
  echo "  issue_coverage_audit=${ISSUE_COVERAGE_AUDIT_PATH}"
  echo
  echo "[xgbv4-post] planned stages:"
  if [[ "${CONTINUE_POST_READINESS_RUN}" == "true" ]]; then
    echo "  1. confirm 7-day readiness"
    echo "  2. reuse existing snapshot, training, eval, model, backtest, and serving artifacts"
  else
    echo "  1. confirm 7-day readiness and refresh settled labels"
    echo "  2. snapshot incumbent champion and fallback artifacts"
    echo "  3. assemble UP 5/1/1 training dataset and DOWN validation dataset from ${LIVE_ROOT}"
    echo "  4. evaluate incumbent and train/calibrate/evaluate fresh xgboost-v4"
    echo "  5. write dataset stability, offline rerun report, and feature-ablation evidence"
    echo "  6. run incumbent, candidate, and DOWN-side direct model backtests"
    echo "  7. run serving-readiness evidence"
  fi
  if [[ "${RUN_SHADOW}" == "true" ]]; then
    echo "  8. run shadow-v1 from ${SHADOW_SINCE_MS:-<missing>} to ${SHADOW_UNTIL_MS:-<missing>} and bootstrap (min ${MIN_SHADOW_SESSION_SECONDS}s)"
  else
    echo "  8. skip shadow/bootstrap; final audits will remain blocked at Stage 3+"
  fi
  if [[ "${RUN_CUTOVER_REPORT}" == "true" ]]; then
    echo "  9. build drift baseline and cutover report from smoke=${SMOKE_PATH} plus GitHub closures=${GITHUB_ISSUE_CLOSURES_PATH}"
  else
    echo "  9. skip cutover report; final audits will remain blocked at Stage 5"
  fi
  echo "  10. run fail-closed champion-promotion, objective, and issue-coverage audits"
}

write_run_manifest() {
  local phase="$1"
  local readiness_json="$2"

  "${PYTHON_BIN}" - "${phase}" "${RUN_MANIFEST_PATH}" "${readiness_json}" \
    "repo_root=${REPO_ROOT}" \
    "live_root=${LIVE_ROOT}" \
    "status_path=${STATUS_PATH}" \
    "manifest_path=${MANIFEST_PATH}" \
    "log_dir=${LOG_DIR}" \
    "screen_session=${SCREEN_SESSION}" \
    "monitoring_db_path=${MONITORING_DB_PATH}" \
    "model_family=${MODEL_FAMILY}" \
    "environment=${ENVIRONMENT}" \
    "run_id=${RUN_ID}" \
    "run_root=${RUN_ROOT}" \
    "force_post_readiness_rerun=${FORCE_POST_READINESS_RERUN}" \
    "continue_post_readiness_run=${CONTINUE_POST_READINESS_RUN}" \
    "post_readiness_sentinel_path=${POST_READINESS_SENTINEL_PATH}" \
    "post_readiness_lock_dir=${POST_READINESS_LOCK_DIR}" \
    "post_readiness_latest_path=${POST_READINESS_LATEST_PATH}" \
    "slack_delivery_status_path=${SLACK_DELIVERY_STATUS_PATH}" \
    "artifact_root=${ARTIFACT_ROOT}" \
    "training_dir=${TRAINING_DIR}" \
    "training_down_dir=${TRAINING_DOWN_DIR}" \
    "model_root=${MODEL_ROOT}" \
    "backtest_root=${BACKTEST_ROOT}" \
    "shadow_root=${SHADOW_ROOT}" \
    "bootstrap_root=${BOOTSTRAP_ROOT}" \
    "cutover_root=${CUTOVER_ROOT}" \
    "audit_root=${AUDIT_ROOT}" \
    "snapshot_path=${SNAPSHOT_PATH}" \
    "objective_audit_path=${OBJECTIVE_AUDIT_PATH}" \
    "issue_coverage_audit_path=${ISSUE_COVERAGE_AUDIT_PATH}" \
    "shadow_report_path=${SHADOW_REPORT_PATH}" \
    "shadow_evaluation_path=${SHADOW_EVALUATION_PATH}" \
    "bootstrap_decision_path=${BOOTSTRAP_DECISION_PATH}" \
    "drift_baseline_path=${DRIFT_BASELINE_PATH}" \
    "cutover_report_path=${CUTOVER_REPORT_PATH}" \
    "smoke_path=${SMOKE_PATH}" \
    "github_issue_closures_path=${GITHUB_ISSUE_CLOSURES_PATH}" \
    "promotion_process_path=${PROMOTION_PROCESS_PATH}" \
    "promotion_repo_runbook_path=${PROMOTION_REPO_RUNBOOK_PATH}" \
    "ensemble_seeds=${ENSEMBLE_SEEDS}" \
    "thresholds=${THRESHOLDS}" \
    "fee_bps=${FEE_BPS}" \
    "slippage_bps=${SLIPPAGE_BPS}" \
    "latency_ms=${LATENCY_MS}" \
    "serving_sample_size=${SERVING_SAMPLE_SIZE}" \
    "serving_batch_sizes=${SERVING_BATCH_SIZES}" \
    "run_shadow=${RUN_SHADOW}" \
    "shadow_since_ms=${SHADOW_SINCE_MS}" \
    "shadow_until_ms=${SHADOW_UNTIL_MS}" \
    "shadow_edge_threshold=${SHADOW_EDGE_THRESHOLD}" \
    "min_shadow_session_seconds=${MIN_SHADOW_SESSION_SECONDS}" \
    "run_cutover_report=${RUN_CUTOVER_REPORT}" \
    "strict_final_audit=${STRICT_FINAL_AUDIT}" <<'PY'
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

phase, output_path, readiness_raw = sys.argv[1:4]
values = dict(arg.split("=", 1) for arg in sys.argv[4:])
try:
    readiness = json.loads(readiness_raw)
except json.JSONDecodeError:
    readiness = {"raw": readiness_raw}

promotion_audit_path = Path(values["audit_root"]) / "champion_promotion_audit.json"
objective_audit_path = Path(values["objective_audit_path"])
issue_coverage_audit_path = Path(values["issue_coverage_audit_path"])


def read_json_dict(path):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def earliest_failed_stage(payload):
    stages = payload.get("stages") if isinstance(payload, dict) else None
    if not isinstance(stages, list):
        return None
    for stage in stages:
        if isinstance(stage, dict) and stage.get("passed") is not True:
            return stage.get("name")
    return None


def bool_or_none(payload, key):
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    return value if isinstance(value, bool) else None


def list_or_empty(value) -> list:
    return value if isinstance(value, list) else []


def dict_or_empty(value) -> dict:
    return value if isinstance(value, dict) else {}


def nested_dict(payload, *keys) -> dict:
    value = payload
    for key in keys:
        if not isinstance(value, dict):
            return {}
        value = value.get(key)
    return value if isinstance(value, dict) else {}


def live_status_summary(payload) -> dict:
    if not isinstance(payload, dict):
        return {"exists": False}
    readiness = nested_dict(payload, "collection_readiness")
    quarantine = nested_dict(payload, "raw_segment_quarantine")
    clean_window = nested_dict(readiness, "quarantine_clean_window")
    feature = nested_dict(readiness, "features_15m_v1")
    labels = nested_dict(readiness, "labels_15m_v1")
    health = nested_dict(payload, "health_evidence")
    return {
        "exists": True,
        "generated_at": payload.get("generated_at"),
        "live_root": payload.get("live_root"),
        "warehouse": payload.get("warehouse"),
        "screen_session": payload.get("screen_session"),
        "screen_state": payload.get("screen_state"),
        "raw_segment_count": payload.get("raw_segment_count"),
        "processed_manifest_rows": payload.get("processed_manifest_rows"),
        "ready_for_training": bool_or_none(readiness, "ready_for_training"),
        "estimated_ready_at": readiness.get("estimated_ready_at"),
        "feature_progress_pct": feature.get("target_progress_pct"),
        "feature_remaining_target_days": feature.get("remaining_target_days"),
        "feature_limiting_family": feature.get("limiting_family"),
        "label_progress_pct": labels.get("target_progress_pct"),
        "label_remaining_target_days": labels.get("remaining_target_days"),
        "label_limiting_family": labels.get("limiting_family"),
        "quarantined_raw_segments": quarantine.get("quarantined_count"),
        "latest_quarantined_segment": quarantine.get("latest_quarantined_segment"),
        "quarantine_clean_window_ready": bool_or_none(clean_window, "meets_target"),
        "quarantine_clean_window_progress_pct": clean_window.get("target_progress_pct"),
        "quarantine_clean_window_remaining_target_days": clean_window.get(
            "remaining_target_days"
        ),
        "quarantine_clean_window_estimated_ready_at": clean_window.get("estimated_ready_at"),
        "unrecovered_error_match_count": health.get("unrecovered_error_match_count"),
    }


def write_json_atomic(target: Path, payload: dict) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(target)
    finally:
        if tmp.exists():
            tmp.unlink()


promotion_audit = read_json_dict(promotion_audit_path)
objective_audit = read_json_dict(objective_audit_path)
issue_coverage_audit = read_json_dict(issue_coverage_audit_path)
live_status = read_json_dict(Path(values["status_path"]))

manifest = {
    "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    "phase": phase,
    "readiness": readiness,
    "live_status_summary": live_status_summary(live_status),
    "inputs": {
        "repo_root": values["repo_root"],
        "live_root": values["live_root"],
        "status_path": values["status_path"],
        "processed_manifest_path": values["manifest_path"],
        "log_dir": values["log_dir"],
        "screen_session": values["screen_session"],
        "monitoring_db_path": values["monitoring_db_path"],
        "model_family": values["model_family"],
        "environment": values["environment"],
    },
    "settings": {
        "run_id": values["run_id"],
        "force_post_readiness_rerun": values["force_post_readiness_rerun"] == "true",
        "continue_post_readiness_run": values["continue_post_readiness_run"] == "true",
        "ensemble_seeds": values["ensemble_seeds"],
        "thresholds": values["thresholds"],
        "fee_bps": values["fee_bps"],
        "slippage_bps": values["slippage_bps"],
        "latency_ms": values["latency_ms"],
        "serving_sample_size": values["serving_sample_size"],
        "serving_batch_sizes": values["serving_batch_sizes"],
        "run_shadow": values["run_shadow"] == "true",
        "shadow_since_ms": values["shadow_since_ms"] or None,
        "shadow_until_ms": values["shadow_until_ms"] or None,
        "shadow_edge_threshold": values["shadow_edge_threshold"],
        "min_shadow_session_seconds": int(values["min_shadow_session_seconds"]),
        "run_cutover_report": values["run_cutover_report"] == "true",
        "strict_final_audit": values["strict_final_audit"] == "true",
    },
    "paths": {
        "run_root": values["run_root"],
        "post_readiness_sentinel_path": values["post_readiness_sentinel_path"],
        "post_readiness_lock_dir": values["post_readiness_lock_dir"],
        "post_readiness_latest_path": values["post_readiness_latest_path"],
        "slack_delivery_status_path": values["slack_delivery_status_path"],
        "promotion_process_path": values["promotion_process_path"],
        "promotion_repo_runbook_path": values["promotion_repo_runbook_path"],
        "artifact_root": values["artifact_root"],
        "training_dir": values["training_dir"],
        "training_down_dir": values["training_down_dir"],
        "model_root": values["model_root"],
        "candidate_model_path": str(Path(values["model_root"]) / "xgboost-v4" / "model.json"),
        "candidate_calibration_path": str(
            Path(values["model_root"]) / "xgboost-v4-calibration" / "calibration.json"
        ),
        "baseline_eval_dir": str(Path(values["model_root"]) / "incumbent-same-dataset"),
        "candidate_eval_dir": str(Path(values["model_root"]) / "xgboost-v4-same-dataset"),
        "dataset_stability_report_path": str(
            Path(values["artifact_root"]) / "dataset-stability" / "dataset_stability_report.json"
        ),
        "offline_rerun_report_path": str(Path(values["artifact_root"]) / "rerun_report.md"),
        "feature_ablation_path": str(
            Path(values["model_root"]) / "xgboost-v4-feature-ablation" / "feature_ablation.json"
        ),
        "candidate_backtest_summary_path": str(
            Path(values["backtest_root"]) / "xgboost-v4" / "summary.json"
        ),
        "down_validation_path": str(
            Path(values["backtest_root"]) / "xgboost-v4-down" / "diagnostics.json"
        ),
        "serving_readiness_path": str(
            Path(values["model_root"]) / "xgboost-v4-serving-readiness.json"
        ),
        "shadow_root": values["shadow_root"],
        "shadow_report_path": values["shadow_report_path"],
        "shadow_evaluation_path": values["shadow_evaluation_path"],
        "bootstrap_root": values["bootstrap_root"],
        "bootstrap_decision_path": values["bootstrap_decision_path"],
        "cutover_root": values["cutover_root"],
        "drift_baseline_path": values["drift_baseline_path"],
        "cutover_report_path": values["cutover_report_path"],
        "audit_root": values["audit_root"],
        "promotion_audit_path": str(promotion_audit_path),
        "snapshot_path": values["snapshot_path"],
        "objective_audit_path": values["objective_audit_path"],
        "issue_coverage_audit_path": values["issue_coverage_audit_path"],
        "smoke_path": values["smoke_path"],
        "github_issue_closures_path": values["github_issue_closures_path"],
    },
    "audit_results": {
        "promotion": {
            "path": str(promotion_audit_path),
            "exists": promotion_audit is not None,
            "decision": promotion_audit.get("decision") if promotion_audit else None,
            "passed": bool_or_none(promotion_audit, "passed"),
            "earliest_failed_stage": earliest_failed_stage(promotion_audit),
        },
        "objective": {
            "path": str(objective_audit_path),
            "exists": objective_audit is not None,
            "decision": objective_audit.get("decision") if objective_audit else None,
            "objective_complete": bool_or_none(objective_audit, "objective_complete"),
            "restatement": dict_or_empty(
                objective_audit.get("objective_restatement") if objective_audit else None
            ),
            "success_criteria": list_or_empty(
                objective_audit.get("objective_success_criteria") if objective_audit else None
            ),
            "blockers": list_or_empty(objective_audit.get("blockers") if objective_audit else None),
            "prompt_to_artifact_blockers": list_or_empty(
                objective_audit.get("prompt_to_artifact_blockers") if objective_audit else None
            ),
            "prompt_to_artifact_checklist": list_or_empty(
                objective_audit.get("prompt_to_artifact_checklist") if objective_audit else None
            ),
            "promotion": dict_or_empty(objective_audit.get("promotion") if objective_audit else None),
        },
        "issue_coverage": {
            "path": str(issue_coverage_audit_path),
            "exists": issue_coverage_audit is not None,
            "generated_at": (
                issue_coverage_audit.get("generated_at") if issue_coverage_audit else None
            ),
            "decision": nested_dict(issue_coverage_audit, "summary").get("decision"),
            "objective_complete": bool_or_none(
                nested_dict(issue_coverage_audit, "summary"),
                "objective_complete",
            ),
            "blocker_count": nested_dict(issue_coverage_audit, "summary").get(
                "blocker_count"
            ),
            "issue_checks": dict_or_empty(
                issue_coverage_audit.get("issue_checks") if issue_coverage_audit else None
            ),
            "objective_success_criteria": dict_or_empty(
                issue_coverage_audit.get("objective_success_criteria")
                if issue_coverage_audit
                else None
            ),
        },
    },
}
path = Path(output_path)
write_json_atomic(path, manifest)
print(f"[xgbv4-post] run manifest={path}")
PY
}

write_completion_sentinel() {
  "${PYTHON_BIN}" - "${POST_READINESS_SENTINEL_PATH}" \
    "latest_path=${POST_READINESS_LATEST_PATH}" \
    "run_id=${RUN_ID}" \
    "run_root=${RUN_ROOT}" \
    "run_manifest_path=${RUN_MANIFEST_PATH}" \
    "promotion_audit_path=${AUDIT_ROOT}/champion_promotion_audit.json" \
    "objective_audit_path=${OBJECTIVE_AUDIT_PATH}" \
    "issue_coverage_audit_path=${ISSUE_COVERAGE_AUDIT_PATH}" <<'PY'
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

path = Path(sys.argv[1])
values = dict(arg.split("=", 1) for arg in sys.argv[2:])


def read_json_dict(raw_path: str) -> dict:
    try:
        payload = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def list_or_empty(value) -> list:
    return value if isinstance(value, list) else []


def dict_or_empty(value) -> dict:
    return value if isinstance(value, dict) else {}


def write_json_atomic(target: Path, payload: dict) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(target)
    finally:
        if tmp.exists():
            tmp.unlink()


promotion = read_json_dict(values["promotion_audit_path"])
objective = read_json_dict(values["objective_audit_path"])
issue_coverage = read_json_dict(values["issue_coverage_audit_path"])
manifest = read_json_dict(values["run_manifest_path"])
payload = {
    "completed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    "completion_scope": "post_readiness_runner_completed",
    "completion_note": (
        "Runner completed fresh post-readiness evidence; objective may remain blocked "
        "until shadow, bootstrap, cutover, and final promotion gates pass."
    ),
    "run_id": values["run_id"],
    "run_root": values["run_root"],
    "run_manifest_path": values["run_manifest_path"],
    "run_manifest_phase": manifest.get("phase"),
    "live_status_summary": dict_or_empty(manifest.get("live_status_summary")),
    "artifact_paths": dict_or_empty(manifest.get("paths")),
    "promotion_audit_path": values["promotion_audit_path"],
    "promotion_decision": promotion.get("decision"),
    "promotion_passed": promotion.get("passed") if isinstance(promotion.get("passed"), bool) else None,
    "objective_audit_path": values["objective_audit_path"],
    "objective_decision": objective.get("decision"),
    "objective_complete": (
        objective.get("objective_complete")
        if isinstance(objective.get("objective_complete"), bool)
        else None
    ),
    "objective_restatement": dict_or_empty(
        objective.get("objective_restatement")
    ),
    "objective_success_criteria": list_or_empty(
        objective.get("objective_success_criteria")
    ),
    "objective_blockers": list_or_empty(objective.get("blockers")),
    "objective_prompt_to_artifact_blockers": list_or_empty(
        objective.get("prompt_to_artifact_blockers")
    ),
    "objective_prompt_to_artifact_checklist": list_or_empty(
        objective.get("prompt_to_artifact_checklist")
    ),
    "objective_promotion": dict_or_empty(objective.get("promotion")),
    "issue_coverage_audit_path": values["issue_coverage_audit_path"],
    "issue_coverage_generated_at": issue_coverage.get("generated_at"),
    "issue_coverage_issue_checks": dict_or_empty(issue_coverage.get("issue_checks")),
    "issue_coverage_objective_success_criteria": dict_or_empty(
        issue_coverage.get("objective_success_criteria")
    ),
    "sentinel_path": str(path),
}
write_json_atomic(path, payload)
latest_path = Path(values["latest_path"])
write_json_atomic(latest_path, payload)
print(f"[xgbv4-post] completion sentinel={path}")
print(f"[xgbv4-post] latest pointer={latest_path}")
PY
}

refresh_live_status_for_audit() {
  run_quiet env BIGAN_DATA_DIR="${LIVE_ROOT}" \
    "${PYTHON_BIN}" -m bigan.ingestion.__main__ live-collection-status \
    --output-path "${STATUS_PATH}" \
    --manifest-path "${MANIFEST_PATH}" \
    --log-dir "${LOG_DIR}" \
    --screen-session "${SCREEN_SESSION}" \
    --lookback-minutes 10 \
    --labels-disabled \
    --gzip-check-limit 20

  READINESS_JSON="$(
    "${PYTHON_BIN}" -m bigan.ingestion.__main__ live-collection-readiness \
      --status-path "${STATUS_PATH}" \
      --no-fail-on-blocked
  )"
}

run_objective_audit() {
  local audit_fail_args=("$@")
  run "${PYTHON_BIN}" -m bigan.ingestion.__main__ xgboost-v4-objective-audit \
    --live-status-path "${STATUS_PATH}" \
    --promotion-audit-path "${AUDIT_ROOT}/champion_promotion_audit.json" \
    --candidate-model-dir "${MODEL_ROOT}/xgboost-v4" \
    --feature-ablation-path "${MODEL_ROOT}/xgboost-v4-feature-ablation/feature_ablation.json" \
    --stability-report-path "${STABILITY_ROOT}/dataset_stability_report.json" \
    --down-validation-path "${BACKTEST_ROOT}/xgboost-v4-down/diagnostics.json" \
    --slack-delivery-status-path "${SLACK_DELIVERY_STATUS_PATH}" \
    --post-readiness-latest-path "${POST_READINESS_LATEST_PATH}" \
    --output-path "${OBJECTIVE_AUDIT_PATH}" \
    ${audit_fail_args[@]+"${audit_fail_args[@]}"}
}

run_issue_coverage_audit() {
  run "${PYTHON_BIN}" -m bigan.ingestion.__main__ xgboost-v4-issue-coverage-audit \
    --output-path "${ISSUE_COVERAGE_AUDIT_PATH}" \
    --live-status-path "${STATUS_PATH}" \
    --promotion-audit-path "${AUDIT_ROOT}/champion_promotion_audit.json" \
    --objective-audit-path "${OBJECTIVE_AUDIT_PATH}" \
    --no-fail-on-blocked
}

objective_audit_only_waits_on_post_readiness_pointer() {
  "${PYTHON_BIN}" - "${OBJECTIVE_AUDIT_PATH}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    print(f"[xgbv4-post] unable to read objective audit JSON {path}: {exc}", file=sys.stderr)
    raise SystemExit(1)

blockers = payload.get("prompt_to_artifact_blockers")
if not isinstance(blockers, list):
    print(
        "[xgbv4-post] objective audit is missing prompt_to_artifact_blockers",
        file=sys.stderr,
    )
    raise SystemExit(1)

unexpected = [
    str(blocker)
    for blocker in blockers
    if not str(blocker).startswith("post_readiness_latest_pointer:")
]
if unexpected:
    print(
        "[xgbv4-post] strict final audit blocked before sentinel by non-pointer objective blockers:",
        file=sys.stderr,
    )
    for blocker in unexpected:
        print(f"  - {blocker}", file=sys.stderr)
    raise SystemExit(1)
PY
}

strict_final_latest_pointer_complete() {
  "${PYTHON_BIN}" - "${POST_READINESS_LATEST_PATH}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    print(f"[xgbv4-post] unable to read latest pointer JSON {path}: {exc}", file=sys.stderr)
    raise SystemExit(1)

required_issue_ids = ("#54", "#55", "#56", "#57", "#58", "#64", "#65")
required_success_ids = (
    "all_requested_github_issues_satisfied",
    "fresh_xgboost_v4_model_created",
    "beats_current_champion",
    "champion_promotion_gates_passed",
    "hourly_slack_status_active",
    "post_readiness_latest_pointer_valid",
)


def nested_bool(mapping: object, key: str) -> bool:
    if not isinstance(mapping, dict):
        return False
    item = mapping.get(key)
    if not isinstance(item, dict):
        return False
    return item.get("passed") is True


issue_checks = payload.get("issue_coverage_issue_checks")
success_criteria = payload.get("issue_coverage_objective_success_criteria")
problems: list[str] = []
if payload.get("run_manifest_phase") != "completed":
    problems.append(f"run_manifest_phase={payload.get('run_manifest_phase')!r}")
if payload.get("promotion_passed") is not True:
    problems.append(f"promotion_passed={payload.get('promotion_passed')!r}")
if payload.get("objective_complete") is not True:
    problems.append(f"objective_complete={payload.get('objective_complete')!r}")
if payload.get("objective_decision") != "COMPLETE":
    problems.append(f"objective_decision={payload.get('objective_decision')!r}")
for issue_id in required_issue_ids:
    if not nested_bool(issue_checks, issue_id):
        problems.append(f"issue_coverage_issue_checks.{issue_id}.passed is not true")
for criterion_id in required_success_ids:
    if not nested_bool(success_criteria, criterion_id):
        problems.append(
            f"issue_coverage_objective_success_criteria.{criterion_id}.passed is not true"
        )

if problems:
    print("[xgbv4-post] strict final latest pointer is incomplete:", file=sys.stderr)
    for problem in problems:
        print(f"  - {problem}", file=sys.stderr)
    raise SystemExit(1)
PY
}

require_command "${PYTHON_BIN}"
require_file "${MANIFEST_PATH}" "processed manifest"
validate_mode_flags
require_live_root_configuration

echo "[xgbv4-post] repo=${REPO_ROOT}"
echo "[xgbv4-post] live root=${LIVE_ROOT}"
echo "[xgbv4-post] run root=${RUN_ROOT}"
if [[ "${CONTINUE_POST_READINESS_RUN}" == "true" ]]; then
  echo "[xgbv4-post] continuing existing post-readiness run root"
fi

if [[ "${PLAN_ONLY}" == "true" ]]; then
  require_file "${STATUS_PATH}" "live status"
  require_status_matches_live_root
  if ! check_status_disk_headroom; then
    echo "[xgbv4-post] PLAN_ONLY continuing after disk headroom warning; no writes will run" >&2
  fi
  READINESS_JSON="$(
    "${PYTHON_BIN}" -m bigan.ingestion.__main__ live-collection-readiness \
      --status-path "${STATUS_PATH}" \
      --no-fail-on-blocked
  )"
  resolve_shadow_window_if_auto
  print_plan "${READINESS_JSON}"
  exit 0
fi

refresh_live_status_for_audit
resolve_shadow_window_if_auto
require_status_matches_live_root
check_status_disk_headroom
if ! printf '%s\n' "${READINESS_JSON}" | readiness_is_true; then
  echo "[xgbv4-post] corpus is not ready; aborting before retrain" >&2
  exit 1
fi

CLAIM_STATUS=0
claim_post_readiness_run || CLAIM_STATUS=$?
if [[ "${CLAIM_STATUS}" -eq 1 ]]; then
  exit 0
fi
if [[ "${CLAIM_STATUS}" -ne 0 ]]; then
  exit "${CLAIM_STATUS}"
fi

mkdir -p "${ARTIFACT_ROOT}" "${MODEL_ROOT}" "${BACKTEST_ROOT}"
if [[ "${CONTINUE_POST_READINESS_RUN}" == "true" ]]; then
  write_run_manifest "continuation_started" "${READINESS_JSON}"
  require_file "${SNAPSHOT_PATH}" "incumbent snapshot"
  INCUMBENT_MODEL_PATH="$(json_get "${SNAPSHOT_PATH}" registry_champion.artifact_uri)"
  INCUMBENT_MODEL_VERSION="$(json_get "${SNAPSHOT_PATH}" registry_champion.model_version)"
  INCUMBENT_CALIBRATION_PATH="$(json_get "${SNAPSHOT_PATH}" registry_champion.calibration_artifact_uri)"
  FALLBACK_MODEL_VERSION="$(json_get "${SNAPSHOT_PATH}" online_model.rollback_to_version)"
  FALLBACK_MODEL_PATH="$(json_get "${SNAPSHOT_PATH}" fallback_registry_model.artifact_uri)"
  if [[ -z "${FALLBACK_MODEL_VERSION}" ]]; then
    FALLBACK_MODEL_VERSION="xgboost-v3"
  fi
  require_file "${INCUMBENT_MODEL_PATH}" "incumbent model artifact"
  require_file "${INCUMBENT_CALIBRATION_PATH}" "incumbent calibration artifact"
  require_file "${FALLBACK_MODEL_PATH}" "fallback model artifact"
  require_file "${TRAINING_DIR}/manifest.json" "training manifest"
  require_file "${TRAINING_DOWN_DIR}/manifest.json" "DOWN validation training manifest"
  require_file "${MODEL_ROOT}/incumbent-same-dataset/manifest.json" "incumbent same-dataset eval"
  require_file "${MODEL_ROOT}/xgboost-v4/model.json" "candidate model"
  require_file "${MODEL_ROOT}/xgboost-v4/feature_schema.json" "candidate feature schema"
  require_file "${MODEL_ROOT}/xgboost-v4-calibration/calibration.json" "candidate calibration"
  require_file "${MODEL_ROOT}/xgboost-v4-same-dataset/manifest.json" "candidate same-dataset eval"
  require_file "${MODEL_ROOT}/xgboost-v4-same-dataset/offline_reference.json" "candidate offline reference"
  require_file "${STABILITY_ROOT}/dataset_stability_report.json" "dataset stability report"
  require_file "${ARTIFACT_ROOT}/rerun_report.md" "offline rerun report"
  require_file "${MODEL_ROOT}/xgboost-v4-feature-ablation/feature_ablation.json" "feature ablation report"
  require_file "${BACKTEST_ROOT}/incumbent/summary.json" "incumbent backtest summary"
  require_file "${BACKTEST_ROOT}/xgboost-v4/summary.json" "candidate backtest summary"
  require_file "${BACKTEST_ROOT}/xgboost-v4-down/diagnostics.json" "DOWN-side validation diagnostics"
  require_file "${MODEL_ROOT}/xgboost-v4-serving-readiness.json" "serving readiness"
else
  write_run_manifest "started" "${READINESS_JSON}"

  run env BIGAN_DATA_DIR="${LIVE_ROOT}" \
    "${PYTHON_BIN}" -m bigan.ingestion.__main__ labels-15m-v1 \
    --monitoring-db-path "${MONITORING_DB_PATH}" \
    --monitoring-model-version xgboost-v4 \
    --request-timeout-seconds "${LABEL_REQUEST_TIMEOUT_SECONDS}" \
    --request-concurrency "${LABEL_REQUEST_CONCURRENCY}" \
    --lookback-minutes "${LABEL_LOOKBACK_MINUTES}" \
    --fee-bps 0 \
    --skip-existing-labels

  run env BIGAN_DATA_DIR="${LIVE_ROOT}" \
    "${PYTHON_BIN}" -m bigan.ingestion.__main__ live-collection-status \
    --output-path "${STATUS_PATH}" \
    --manifest-path "${MANIFEST_PATH}" \
    --log-dir "${LOG_DIR}" \
    --screen-session "${SCREEN_SESSION}" \
    --lookback-minutes 10 \
    --labels-disabled \
    --gzip-check-limit 20

  READINESS_JSON="$(
    "${PYTHON_BIN}" -m bigan.ingestion.__main__ live-collection-readiness \
      --status-path "${STATUS_PATH}" \
      --no-fail-on-blocked
  )"
  require_status_matches_live_root
  check_status_disk_headroom
  if ! printf '%s\n' "${READINESS_JSON}" | readiness_is_true; then
    write_run_manifest "post_label_readiness_blocked" "${READINESS_JSON}"
    echo "[xgbv4-post] corpus lost readiness after label refresh; aborting" >&2
    exit 1
  fi
  write_run_manifest "post_label_readiness_confirmed" "${READINESS_JSON}"

  run "${PYTHON_BIN}" -m bigan.ingestion.__main__ champion-state-snapshot-v1 \
    --output-path "${SNAPSHOT_PATH}" \
    --monitoring-db-path "${MONITORING_DB_PATH}" \
    --model-family "${MODEL_FAMILY}" \
    --environment "${ENVIRONMENT}"

  INCUMBENT_MODEL_PATH="$(json_get "${SNAPSHOT_PATH}" registry_champion.artifact_uri)"
  INCUMBENT_MODEL_VERSION="$(json_get "${SNAPSHOT_PATH}" registry_champion.model_version)"
  INCUMBENT_CALIBRATION_PATH="$(json_get "${SNAPSHOT_PATH}" registry_champion.calibration_artifact_uri)"
  FALLBACK_MODEL_VERSION="$(json_get "${SNAPSHOT_PATH}" online_model.rollback_to_version)"
  FALLBACK_MODEL_PATH="$(json_get "${SNAPSHOT_PATH}" fallback_registry_model.artifact_uri)"

  if [[ -z "${FALLBACK_MODEL_VERSION}" ]]; then
    FALLBACK_MODEL_VERSION="xgboost-v3"
  fi
  require_file "${INCUMBENT_MODEL_PATH}" "incumbent model artifact"
  require_file "${INCUMBENT_CALIBRATION_PATH}" "incumbent calibration artifact"
  require_file "${FALLBACK_MODEL_PATH}" "fallback model artifact"

  run env BIGAN_DATA_DIR="${LIVE_ROOT}" \
    "${PYTHON_BIN}" -m bigan.ingestion.__main__ training-dataset-v1 \
    --output-dir "${TRAINING_DIR}" \
    --train-fraction 0.7142857143 \
    --val-fraction 0.1428571429 \
    --outcome-side UP

  run env BIGAN_DATA_DIR="${LIVE_ROOT}" \
    "${PYTHON_BIN}" -m bigan.ingestion.__main__ training-dataset-v1 \
    --output-dir "${TRAINING_DOWN_DIR}" \
    --train-fraction 0.7142857143 \
    --val-fraction 0.1428571429 \
    --outcome-side DOWN

  run "${PYTHON_BIN}" -m bigan.ingestion.__main__ model-eval-v1 \
    --model-path "${INCUMBENT_MODEL_PATH}" \
    --calibration-path "${INCUMBENT_CALIBRATION_PATH}" \
    --dataset-dir "${TRAINING_DIR}" \
    --output-dir "${MODEL_ROOT}/incumbent-same-dataset"

  run "${PYTHON_BIN}" -m bigan.ingestion.__main__ xgboost-v4 \
    --dataset-dir "${TRAINING_DIR}" \
    --output-dir "${MODEL_ROOT}/xgboost-v4" \
    --ensemble-seeds "${ENSEMBLE_SEEDS}"

  run "${PYTHON_BIN}" -m bigan.ingestion.__main__ calibration-v1 \
    --model-path "${MODEL_ROOT}/xgboost-v4/model.json" \
    --dataset-dir "${TRAINING_DIR}" \
    --output-dir "${MODEL_ROOT}/xgboost-v4-calibration"

  run "${PYTHON_BIN}" -m bigan.ingestion.__main__ model-eval-v1 \
    --model-path "${MODEL_ROOT}/xgboost-v4/model.json" \
    --calibration-path "${MODEL_ROOT}/xgboost-v4-calibration/calibration.json" \
    --dataset-dir "${TRAINING_DIR}" \
    --output-dir "${MODEL_ROOT}/xgboost-v4-same-dataset"

  run "${PYTHON_BIN}" -m bigan.ingestion.__main__ dataset-stability-report-v1 \
    --dataset-dir "${TRAINING_DIR}" \
    --output-dir "${STABILITY_ROOT}"

  run "${PYTHON_BIN}" -m bigan.ingestion.__main__ offline-rerun-report-v1 \
    --baseline-eval-dir "${MODEL_ROOT}/incumbent-same-dataset" \
    --candidate-eval-dir "${MODEL_ROOT}/xgboost-v4-same-dataset" \
    --output-path "${ARTIFACT_ROOT}/rerun_report.md"

  run "${PYTHON_BIN}" -m bigan.ingestion.__main__ feature-ablation-report-v1 \
    --model-path "${MODEL_ROOT}/xgboost-v4/model.json" \
    --calibration-path "${MODEL_ROOT}/xgboost-v4-calibration/calibration.json" \
    --dataset-dir "${TRAINING_DIR}" \
    --output-dir "${MODEL_ROOT}/xgboost-v4-feature-ablation" \
    --split test

  run env BIGAN_DATA_DIR="${LIVE_ROOT}" \
    "${PYTHON_BIN}" -m bigan.ingestion.__main__ backtest-model-v1 \
    --model-path "${INCUMBENT_MODEL_PATH}" \
    --calibration-path "${INCUMBENT_CALIBRATION_PATH}" \
    --dataset-dir "${TRAINING_DIR}" \
    --output-dir "${BACKTEST_ROOT}/incumbent" \
    --model-version "${INCUMBENT_MODEL_VERSION}" \
    --thresholds "${THRESHOLDS}" \
    --fee-bps "${FEE_BPS}" \
    --slippage-bps "${SLIPPAGE_BPS}" \
    --latency-ms "${LATENCY_MS}"

  run env BIGAN_DATA_DIR="${LIVE_ROOT}" \
    "${PYTHON_BIN}" -m bigan.ingestion.__main__ backtest-model-v1 \
    --model-path "${MODEL_ROOT}/xgboost-v4/model.json" \
    --calibration-path "${MODEL_ROOT}/xgboost-v4-calibration/calibration.json" \
    --dataset-dir "${TRAINING_DIR}" \
    --output-dir "${BACKTEST_ROOT}/xgboost-v4" \
    --model-version xgboost-v4 \
    --thresholds "${THRESHOLDS}" \
    --fee-bps "${FEE_BPS}" \
    --slippage-bps "${SLIPPAGE_BPS}" \
    --latency-ms "${LATENCY_MS}"

  run env BIGAN_DATA_DIR="${LIVE_ROOT}" \
    "${PYTHON_BIN}" -m bigan.ingestion.__main__ backtest-model-v1 \
    --model-path "${MODEL_ROOT}/xgboost-v4/model.json" \
    --calibration-path "${MODEL_ROOT}/xgboost-v4-calibration/calibration.json" \
    --dataset-dir "${TRAINING_DOWN_DIR}" \
    --output-dir "${BACKTEST_ROOT}/xgboost-v4-down" \
    --model-version xgboost-v4 \
    --thresholds "${THRESHOLDS}" \
    --required-outcome-side DOWN \
    --fee-bps "${FEE_BPS}" \
    --slippage-bps "${SLIPPAGE_BPS}" \
    --latency-ms "${LATENCY_MS}"

  run "${PYTHON_BIN}" -m bigan.ingestion.__main__ serving-readiness-v1 \
    --model-path "${MODEL_ROOT}/xgboost-v4/model.json" \
    --feature-schema-path "${MODEL_ROOT}/xgboost-v4/feature_schema.json" \
    --dataset-dir "${TRAINING_DIR}" \
    --output-path "${MODEL_ROOT}/xgboost-v4-serving-readiness.json" \
    --sample-size "${SERVING_SAMPLE_SIZE}" \
    --batch-sizes "${SERVING_BATCH_SIZES}" \
    --fallback-model-path "${FALLBACK_MODEL_PATH}" \
    --rollback-runbook-path docs/runbooks/model_rollback.md
fi

SHADOW_AUDIT_ARGS=()
if [[ "${RUN_SHADOW}" == "true" ]]; then
  if [[ -z "${SHADOW_SINCE_MS}" || -z "${SHADOW_UNTIL_MS}" ]]; then
    echo "[xgbv4-post] RUN_SHADOW=true requires SHADOW_SINCE_MS and SHADOW_UNTIL_MS" >&2
    exit 1
  fi
  mkdir -p "${SHADOW_ROOT}" "${BOOTSTRAP_ROOT}"
  run env BIGAN_DATA_DIR="${LIVE_ROOT}" \
    "${PYTHON_BIN}" -m bigan.ingestion.__main__ shadow-v1 \
    --champion-model-path "${INCUMBENT_MODEL_PATH}" \
    --challenger-model-path "${MODEL_ROOT}/xgboost-v4/model.json" \
    --champion-calibration-path "${INCUMBENT_CALIBRATION_PATH}" \
    --challenger-calibration-path "${MODEL_ROOT}/xgboost-v4-calibration/calibration.json" \
    --warehouse-dir "${LIVE_ROOT}/warehouse" \
    --output-path "${SHADOW_REPORT_PATH}" \
    --evaluation-output-path "${SHADOW_ROOT}/xgboost-v4-shadow-evaluation.md" \
    --evaluation-json-output-path "${SHADOW_EVALUATION_PATH}" \
    --offline-reference-path "${MODEL_ROOT}/xgboost-v4-same-dataset/offline_reference.json" \
    --edge-threshold "${SHADOW_EDGE_THRESHOLD}" \
    --since-ms "${SHADOW_SINCE_MS}" \
    --until-ms "${SHADOW_UNTIL_MS}"

  run "${PYTHON_BIN}" -m bigan.ingestion.__main__ bootstrap-champion-v1 \
    --baseline-dir "${MODEL_ROOT}/incumbent-same-dataset" \
    --baseline-backtest-summary-path "${BACKTEST_ROOT}/incumbent/summary.json" \
    --candidate-dir "${MODEL_ROOT}/xgboost-v4-same-dataset" \
    --calibration-dir "${MODEL_ROOT}/xgboost-v4-calibration" \
    --candidate-backtest-summary-path "${BACKTEST_ROOT}/xgboost-v4/summary.json" \
    --serving-readiness-path "${MODEL_ROOT}/xgboost-v4-serving-readiness.json" \
    --feature-schema-path "${MODEL_ROOT}/xgboost-v4/feature_schema.json" \
    --model-complexity-notes-path docs/models/xgboost-v4.md \
    --shadow-evaluation-path "${SHADOW_EVALUATION_PATH}" \
    --rollback-runbook-path docs/runbooks/model_rollback.md \
    --output-dir "${BOOTSTRAP_ROOT}" \
    --replace-champion

  SHADOW_AUDIT_ARGS=(
    --shadow-evaluation-path "${SHADOW_EVALUATION_PATH}"
    --bootstrap-decision-path "${BOOTSTRAP_DECISION_PATH}"
  )
elif [[ -f "${SHADOW_EVALUATION_PATH}" && -f "${BOOTSTRAP_DECISION_PATH}" ]]; then
  echo "[xgbv4-post] RUN_SHADOW=false; using existing shadow/bootstrap evidence for audits"
  SHADOW_AUDIT_ARGS=(
    --shadow-evaluation-path "${SHADOW_EVALUATION_PATH}"
    --bootstrap-decision-path "${BOOTSTRAP_DECISION_PATH}"
  )
else
  echo "[xgbv4-post] RUN_SHADOW=false; Stage 3+ remain blocked until full-session shadow evidence exists"
fi

CUTOVER_AUDIT_ARGS=()
if [[ "${RUN_CUTOVER_REPORT}" == "true" ]]; then
  require_file "${SMOKE_PATH}" "cutover smoke artifact"
  require_file "${GITHUB_ISSUE_CLOSURES_PATH}" "GitHub issue closure evidence"
  require_file "${BOOTSTRAP_DECISION_PATH}" "bootstrap decision"
  require_file "${SHADOW_EVALUATION_PATH}" "shadow evaluation"
  mkdir -p "${CUTOVER_ROOT}"
  run "${PYTHON_BIN}" -m bigan.ingestion.__main__ drift-baseline-v1 \
    --offline-reference-path "${MODEL_ROOT}/xgboost-v4-same-dataset/offline_reference.json" \
    --output-path "${DRIFT_BASELINE_PATH}"
  run "${PYTHON_BIN}" -m bigan.ingestion.__main__ champion-cutover-report-v1 \
    --output-path "${CUTOVER_REPORT_PATH}" \
    --monitoring-db-path "${MONITORING_DB_PATH}" \
    --model-family "${MODEL_FAMILY}" \
    --environment "${ENVIRONMENT}" \
    --smoke-path "${SMOKE_PATH}" \
    --drift-baseline-path "${DRIFT_BASELINE_PATH}" \
    --bootstrap-decision-path "${BOOTSTRAP_DECISION_PATH}" \
    --shadow-evaluation-path "${SHADOW_EVALUATION_PATH}" \
    --serving-readiness-path "${MODEL_ROOT}/xgboost-v4-serving-readiness.json" \
    --github-issue-closures-path "${GITHUB_ISSUE_CLOSURES_PATH}"
  CUTOVER_AUDIT_ARGS=(--cutover-report-path "${CUTOVER_REPORT_PATH}")
fi

AUDIT_FAIL_ARGS=(--no-fail-on-blocked)
if [[ "${STRICT_FINAL_AUDIT}" == "true" ]]; then
  AUDIT_FAIL_ARGS=()
fi

refresh_live_status_for_audit

AUDIT_EXIT=0
if ! run "${PYTHON_BIN}" -m bigan.ingestion.__main__ champion-promotion-audit \
  --promotion-process-path "${PROMOTION_PROCESS_PATH}" \
  --repo-promotion-runbook-path "${PROMOTION_REPO_RUNBOOK_PATH}" \
  --live-status-path "${STATUS_PATH}" \
  --offline-rerun-report-path "${ARTIFACT_ROOT}/rerun_report.md" \
  --baseline-eval-dir "${MODEL_ROOT}/incumbent-same-dataset" \
  --candidate-eval-dir "${MODEL_ROOT}/xgboost-v4-same-dataset" \
  --baseline-backtest-summary-path "${BACKTEST_ROOT}/incumbent/summary.json" \
  --candidate-backtest-summary-path "${BACKTEST_ROOT}/xgboost-v4/summary.json" \
  ${SHADOW_AUDIT_ARGS[@]+"${SHADOW_AUDIT_ARGS[@]}"} \
  --serving-readiness-path "${MODEL_ROOT}/xgboost-v4-serving-readiness.json" \
  ${CUTOVER_AUDIT_ARGS[@]+"${CUTOVER_AUDIT_ARGS[@]}"} \
  --rollback-runbook-path docs/runbooks/model_rollback.md \
  --expected-fallback-model-version "${FALLBACK_MODEL_VERSION}" \
  --output-dir "${AUDIT_ROOT}" \
  ${AUDIT_FAIL_ARGS[@]+"${AUDIT_FAIL_ARGS[@]}"}; then
  AUDIT_EXIT=1
fi

if ! run_objective_audit --no-fail-on-blocked; then
  AUDIT_EXIT=1
elif [[ "${STRICT_FINAL_AUDIT}" == "true" ]] && ! objective_audit_only_waits_on_post_readiness_pointer; then
  AUDIT_EXIT=1
fi

if ! run_issue_coverage_audit; then
  AUDIT_EXIT=1
fi

if [[ "${AUDIT_EXIT}" -ne 0 ]]; then
  write_run_manifest "audit_blocked" "${READINESS_JSON}"
  echo "[xgbv4-post] final audits are blocked; see ${RUN_MANIFEST_PATH}" >&2
  exit "${AUDIT_EXIT}"
fi

write_run_manifest "completed" "${READINESS_JSON}"
write_completion_sentinel
if ! run_objective_audit ${AUDIT_FAIL_ARGS[@]+"${AUDIT_FAIL_ARGS[@]}"}; then
  echo "[xgbv4-post] objective audit failed after writing post-readiness sentinel" >&2
  exit 1
fi
if ! run_issue_coverage_audit; then
  echo "[xgbv4-post] issue coverage audit failed after writing post-readiness sentinel" >&2
  exit 1
fi
write_run_manifest "completed" "${READINESS_JSON}"
write_completion_sentinel
if [[ "${STRICT_FINAL_AUDIT}" == "true" ]]; then
  strict_final_latest_pointer_complete
fi

echo
echo "[xgbv4-post] complete; run root: ${RUN_ROOT}"
