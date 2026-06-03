#!/bin/bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Check operational risk for the active xgboost-v4 clean atomic collection.

This script is intentionally non-destructive: it reads the current live status
artifact, prints the collection readiness and disk-headroom evidence, and lists
large local reclaim candidates for a human to review. It never prunes Docker,
deletes simulator data, removes caches, or edits repo artifacts.

Environment overrides:
  STATUS_PATH       live-collection-status JSON path.
  LIVE_ROOT         Clean xgboost-v4 live corpus root.
  STATUS_MAX_AGE_SECONDS
                    Max age before the status artifact is marked stale. Default: 1800.
  SHOW_LIVE_ROOTS   true to include local xgboost-v4 live-root sizes. Default: true

Options:
  --json            emit machine-readable JSON instead of human-readable text.
  --output-path PATH
                    write the JSON payload to PATH atomically. Implies --json.

Exit codes:
  0  status readable and disk headroom is not blocked
  1  missing dependency, missing status file, or invalid JSON
  2  disk headroom is blocked in the status artifact
EOF
}

OUTPUT_FORMAT="human"
OUTPUT_PATH=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --json)
      OUTPUT_FORMAT="json"
      ;;
    --output-path)
      if [[ $# -lt 2 ]]; then
        echo "missing value for --output-path" >&2
        usage >&2
        exit 1
      fi
      OUTPUT_FORMAT="json"
      OUTPUT_PATH="$2"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
  shift
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

STATUS_PATH="${STATUS_PATH:-data/xgboost-v4-run-20260523T103814Z/artifacts/live_multimarket_7d_collection_status_latest.json}"
LIVE_ROOT="${LIVE_ROOT:-data/live/xgboost-v4-multimarket-7d-atomic-20260523T125657Z}"
STATUS_MAX_AGE_SECONDS="${STATUS_MAX_AGE_SECONDS:-1800}"
SHOW_LIVE_ROOTS="${SHOW_LIVE_ROOTS:-true}"

if ! command -v jq >/dev/null 2>&1; then
  echo "missing required dependency: jq" >&2
  exit 1
fi

if [[ ! -f "${STATUS_PATH}" ]]; then
  echo "missing status artifact: ${STATUS_PATH}" >&2
  exit 1
fi

if ! jq empty "${STATUS_PATH}" >/dev/null; then
  echo "status artifact is not valid JSON: ${STATUS_PATH}" >&2
  exit 1
fi

format_gib() {
  local bytes="${1:-}"
  if [[ -z "${bytes}" || "${bytes}" == "null" ]]; then
    printf 'n/a'
    return
  fi
  awk -v bytes="${bytes}" 'BEGIN { printf "%.2fGiB", bytes / 1073741824 }'
}

write_json_output() {
  local output_path="$1"
  local payload="$2"
  local output_dir
  local tmp_path

  output_dir="$(dirname "${output_path}")"
  if [[ "${output_dir}" != "." ]]; then
    mkdir -p "${output_dir}"
  fi
  tmp_path="${output_path}.$$.$RANDOM.tmp"
  printf '%s\n' "${payload}" > "${tmp_path}"
  mv "${tmp_path}" "${output_path}"
}

json_value() {
  local filter="$1"
  jq -r "try (${filter}) catch null | if . == null then \"n/a\" else . end" "${STATUS_PATH}"
}

status_generated_at="$(json_value '.generated_at')"
status_age_seconds="$(
  jq -r '
    def normalize_timestamp:
      tostring
      | sub("\\.[0-9]+Z$"; "Z")
      | sub("\\+00:00$"; "Z");
    try ((now - (.generated_at | normalize_timestamp | fromdateiso8601)) | if . < 0 then 0 else . end | floor)
    catch "n/a"
  ' "${STATUS_PATH}"
)"
status_artifact_fresh="$(
  awk -v age="${status_age_seconds}" -v max_age="${STATUS_MAX_AGE_SECONDS}" '
    BEGIN {
      if (age == "" || age == "n/a" || age == "null" || max_age == "" || max_age == "n/a" || max_age == "null") {
        print "n/a"
      } else if (age + 0 <= max_age + 0) {
        print "true"
      } else {
        print "false"
      }
    }
  '
)"
screen_session="$(json_value '.screen_session')"
screen_state="$(json_value '.screen_state')"
ready_for_training="$(json_value '.collection_readiness.ready_for_training')"
estimated_ready_at="$(json_value '.collection_readiness.estimated_ready_at')"
feature_progress="$(json_value '.collection_readiness.features_15m_v1.target_progress_pct')"
feature_remaining_days="$(json_value '.collection_readiness.features_15m_v1.remaining_target_days')"
feature_limiting_family="$(json_value '.collection_readiness.features_15m_v1.limiting_family')"
label_progress="$(json_value '.collection_readiness.labels_15m_v1.target_progress_pct')"
label_remaining_days="$(json_value '.collection_readiness.labels_15m_v1.remaining_target_days')"
label_limiting_family="$(json_value '.collection_readiness.labels_15m_v1.limiting_family')"
quarantine_progress="$(json_value '.collection_readiness.quarantine_clean_window.target_progress_pct')"
quarantine_remaining_days="$(json_value '.collection_readiness.quarantine_clean_window.remaining_target_days')"
quarantine_ready_at="$(json_value '.collection_readiness.quarantine_clean_window.estimated_ready_at')"
quarantined_count="$(json_value '.collection_readiness.quarantine_clean_window.quarantined_count')"
latest_quarantined_path="$(json_value '.collection_readiness.quarantine_clean_window.latest_quarantined_segment.path')"
raw_fresh="$(json_value '.liveness_evidence.raw_segments_fresh')"
manifest_fresh="$(json_value '.liveness_evidence.processed_manifest_fresh')"
latest_raw_age="$(json_value '.liveness_evidence.latest_raw_segment.age_seconds')"
latest_processed_age="$(json_value '.liveness_evidence.latest_processed_segment.age_seconds')"
invalid_count="$(json_value '.raw_segment_integrity.invalid_count')"
unrecovered_error_count="$(json_value '.health_evidence.unrecovered_error_match_count')"
headroom_ok="$(json_value '.disk_headroom_evidence.headroom_ok')"
headroom_low_margin="$(json_value '.disk_headroom_evidence.headroom_low_margin')"
free_bytes="$(json_value '.disk_headroom_evidence.free_bytes')"
required_free_bytes="$(json_value '.disk_headroom_evidence.required_free_bytes')"
headroom_margin_bytes="$(json_value '.disk_headroom_evidence.headroom_margin_bytes')"
projected_remaining_bytes="$(json_value '.disk_headroom_evidence.projected_remaining_bytes')"
live_root_size_bytes="$(json_value '.disk_headroom_evidence.live_root_size_bytes')"
low_margin_threshold_bytes="$(json_value '.disk_headroom_evidence.low_margin_threshold_bytes')"
min_free_bytes="$(json_value '.disk_headroom_evidence.min_free_bytes')"
observed_raw_span_days="$(json_value '.disk_headroom_evidence.observed_raw_span_days')"
current_fs_free_bytes="n/a"
current_fs_headroom_ok="n/a"
current_fs_low_margin="n/a"
current_fs_margin_bytes="n/a"
current_reclaim_to_clear_block="n/a"
current_reclaim_to_clear_low_margin="n/a"

estimated_growth_bytes_per_day="$(
  awk -v size="${live_root_size_bytes}" -v days="${observed_raw_span_days}" '
    BEGIN {
      if (size == "" || size == "n/a" || size == "null" || days == "" || days == "n/a" || days == "null" || days <= 0) {
        print "n/a"
      } else {
        printf "%.0f", size / days
      }
    }
  '
)"

estimate_days_to_min_free() {
  local free="$1"
  awk -v free="${free}" -v min_free="${min_free_bytes}" -v growth="${estimated_growth_bytes_per_day}" '
    BEGIN {
      if (free == "" || free == "n/a" || free == "null" || min_free == "" || min_free == "n/a" || min_free == "null" || growth == "" || growth == "n/a" || growth == "null" || growth <= 0) {
        print "n/a"
      } else {
        days = (free - min_free) / growth
        if (days < 0) {
          days = 0
        }
        printf "%.3f", days
      }
    }
  '
}

status_days_to_min_free="$(estimate_days_to_min_free "${free_bytes}")"
current_days_to_min_free="n/a"

estimated_days_to_ready="$(
  awk \
    -v features="${feature_remaining_days}" \
    -v labels="${label_remaining_days}" \
    -v quarantine="${quarantine_remaining_days}" '
    function valid(value) {
      return value != "" && value != "n/a" && value != "null"
    }
    BEGIN {
      max_days = -1
      if (valid(features) && features + 0 > max_days) {
        max_days = features + 0
      }
      if (valid(labels) && labels + 0 > max_days) {
        max_days = labels + 0
      }
      if (valid(quarantine) && quarantine + 0 > max_days) {
        max_days = quarantine + 0
      }
      if (max_days < 0) {
        print "n/a"
      } else {
        printf "%.3f", max_days
      }
    }
  '
)"

min_free_before_ready() {
  local days_to_min_free="$1"
  awk -v days_to_min_free="${days_to_min_free}" -v days_to_ready="${estimated_days_to_ready}" '
    BEGIN {
      if (days_to_min_free == "" || days_to_min_free == "n/a" || days_to_min_free == "null" || days_to_ready == "" || days_to_ready == "n/a" || days_to_ready == "null") {
        print "n/a"
      } else if (days_to_min_free + 0 < days_to_ready + 0) {
        print "true"
      } else {
        print "false"
      }
    }
  '
}

status_min_free_before_ready="$(min_free_before_ready "${status_days_to_min_free}")"
current_min_free_before_ready="n/a"

bytes_to_clear_block="$(
  awk -v margin="${headroom_margin_bytes}" '
    BEGIN {
      if (margin == "" || margin == "n/a" || margin == "null") {
        print "n/a"
      } else if (margin < 0) {
        printf "%.0f", -margin
      } else {
        print 0
      }
    }
  '
)"
bytes_to_clear_low_margin="$(
  awk -v margin="${headroom_margin_bytes}" -v threshold="${low_margin_threshold_bytes}" '
    BEGIN {
      if (margin == "" || margin == "n/a" || margin == "null" || threshold == "" || threshold == "n/a" || threshold == "null") {
        print "n/a"
      } else {
        needed = threshold - margin
        if (needed < 0) {
          needed = 0
        }
        printf "%.0f", needed
      }
    }
  '
)"

if [[ -e "${LIVE_ROOT}" ]]; then
  current_fs_free_kib="$(df -Pk "${LIVE_ROOT}" 2>/dev/null | awk 'NR == 2 { print $4 }')"
  if [[ -n "${current_fs_free_kib}" && "${current_fs_free_kib}" =~ ^[0-9]+$ ]]; then
    current_fs_free_bytes="$(( current_fs_free_kib * 1024 ))"
    current_fs_margin_bytes="$(
      awk -v free="${current_fs_free_bytes}" -v required="${required_free_bytes}" '
        BEGIN {
          if (free == "" || free == "n/a" || required == "" || required == "n/a" || required == "null") {
            print "n/a"
          } else {
            printf "%.0f", free - required
          }
        }
      '
    )"
    current_fs_headroom_ok="$(
      awk -v margin="${current_fs_margin_bytes}" '
        BEGIN {
          if (margin == "" || margin == "n/a" || margin == "null") {
            print "n/a"
          } else if (margin >= 0) {
            print "true"
          } else {
            print "false"
          }
        }
      '
    )"
    current_fs_low_margin="$(
      awk -v margin="${current_fs_margin_bytes}" -v threshold="${low_margin_threshold_bytes}" '
        BEGIN {
          if (margin == "" || margin == "n/a" || margin == "null" || threshold == "" || threshold == "n/a" || threshold == "null") {
            print "n/a"
          } else if (margin < threshold) {
            print "true"
          } else {
            print "false"
          }
        }
      '
    )"
    current_reclaim_to_clear_block="$(
      awk -v free="${current_fs_free_bytes}" -v required="${required_free_bytes}" '
        BEGIN {
          if (free == "" || free == "n/a" || required == "" || required == "n/a" || required == "null") {
            print "n/a"
          } else {
            needed = required - free
            if (needed < 0) {
              needed = 0
            }
            printf "%.0f", needed
          }
        }
      '
    )"
    current_reclaim_to_clear_low_margin="$(
      awk -v free="${current_fs_free_bytes}" -v required="${required_free_bytes}" -v threshold="${low_margin_threshold_bytes}" '
        BEGIN {
          if (free == "" || free == "n/a" || required == "" || required == "n/a" || required == "null" || threshold == "" || threshold == "n/a" || threshold == "null") {
            print "n/a"
          } else {
            needed = required + threshold - free
            if (needed < 0) {
              needed = 0
            }
            printf "%.0f", needed
          }
        }
      '
    )"
    current_days_to_min_free="$(estimate_days_to_min_free "${current_fs_free_bytes}")"
    current_min_free_before_ready="$(min_free_before_ready "${current_days_to_min_free}")"
  fi
fi

candidate_paths=(
  "${HOME}/Library/Containers/com.docker.docker|Docker container data"
  "${HOME}/Library/Containers/com.docker.docker/Data/vms|Docker VM store"
  "${HOME}/Library/Developer/CoreSimulator|CoreSimulator devices"
  "${HOME}/Library/Developer/Xcode|Xcode developer data"
  "${HOME}/Library/Caches|user caches"
)
candidate_lines=()
candidate_json_items=()

for entry in "${candidate_paths[@]}"; do
  path="${entry%%|*}"
  label="${entry#*|}"
  if [[ -e "${path}" ]]; then
    if size="$(du -sh "${path}" 2>/dev/null | awk '{print $1}')"; then
      :
    else
      size="unknown"
    fi
    if size_kib="$(du -sk "${path}" 2>/dev/null | awk '{print $1}')"; then
      :
    else
      size_kib="n/a"
    fi
    candidate_lines+=("$(printf '  %-28s %8s  %s' "${label}" "${size:-unknown}" "${path}")")
    candidate_json_items+=("$(
      jq -nc \
        --arg label "${label}" \
        --arg path "${path}" \
        --arg size_human "${size:-unknown}" \
        --arg size_kib "${size_kib}" \
        '{label: $label, path: $path, size_human: $size_human, size_kib: ($size_kib | tonumber?)}'
    )")
  fi
done

candidate_inventory_json="[]"
if [[ "${#candidate_json_items[@]}" -gt 0 ]]; then
  candidate_inventory_json="$(printf '%s\n' "${candidate_json_items[@]}" | jq -s .)"
fi

live_root_lines=()
live_root_json_items=()
if [[ "${SHOW_LIVE_ROOTS}" == "true" && -d data/live ]]; then
  while IFS= read -r -d '' root; do
    if root_size="$(du -sh "${root}" 2>/dev/null | awk '{print $1}')"; then
      :
    else
      root_size="unknown"
    fi
    if root_size_kib="$(du -sk "${root}" 2>/dev/null | awk '{print $1}')"; then
      :
    else
      root_size_kib="n/a"
    fi
    live_root_lines+=("$(printf '  %8s  %s' "${root_size:-unknown}" "${root}")")
    live_root_json_items+=("$(
      jq -nc \
        --arg path "${root}" \
        --arg size_human "${root_size:-unknown}" \
        --arg size_kib "${root_size_kib}" \
        '{path: $path, size_human: $size_human, size_kib: ($size_kib | tonumber?)}'
    )")
  done < <(find data/live -maxdepth 1 -type d -name 'xgboost-v4-*' -print0)
fi

live_roots_json="[]"
if [[ "${#live_root_json_items[@]}" -gt 0 ]]; then
  live_roots_json="$(printf '%s\n' "${live_root_json_items[@]}" | jq -s 'sort_by(.size_kib // 0)')"
fi

script_exit_code=0
status_level="ok"
if [[ "${headroom_ok}" == "false" ]]; then
  script_exit_code=2
  status_level="blocked"
elif [[ "${headroom_low_margin}" == "true" || "${status_artifact_fresh}" == "false" ]]; then
  status_level="warning"
fi

if [[ "${OUTPUT_FORMAT}" == "json" ]]; then
  json_payload="$(
    jq -n \
    --arg status_path "${STATUS_PATH}" \
    --arg generated_at "${status_generated_at}" \
    --arg status_age_seconds "${status_age_seconds}" \
    --arg status_max_age_seconds "${STATUS_MAX_AGE_SECONDS}" \
    --arg status_artifact_fresh "${status_artifact_fresh}" \
    --arg screen_session "${screen_session}" \
    --arg screen_state "${screen_state}" \
    --arg ready_for_training "${ready_for_training}" \
    --arg estimated_ready_at "${estimated_ready_at}" \
    --arg feature_progress "${feature_progress}" \
    --arg feature_remaining_days "${feature_remaining_days}" \
    --arg feature_limiting_family "${feature_limiting_family}" \
    --arg label_progress "${label_progress}" \
    --arg label_remaining_days "${label_remaining_days}" \
    --arg label_limiting_family "${label_limiting_family}" \
    --arg quarantine_progress "${quarantine_progress}" \
    --arg quarantine_remaining_days "${quarantine_remaining_days}" \
    --arg quarantine_ready_at "${quarantine_ready_at}" \
    --arg quarantined_count "${quarantined_count}" \
    --arg latest_quarantined_path "${latest_quarantined_path}" \
    --arg raw_fresh "${raw_fresh}" \
    --arg manifest_fresh "${manifest_fresh}" \
    --arg latest_raw_age "${latest_raw_age}" \
    --arg latest_processed_age "${latest_processed_age}" \
    --arg invalid_count "${invalid_count}" \
    --arg unrecovered_error_count "${unrecovered_error_count}" \
    --arg headroom_ok "${headroom_ok}" \
    --arg headroom_low_margin "${headroom_low_margin}" \
    --arg free_bytes "${free_bytes}" \
    --arg required_free_bytes "${required_free_bytes}" \
    --arg projected_remaining_bytes "${projected_remaining_bytes}" \
    --arg headroom_margin_bytes "${headroom_margin_bytes}" \
    --arg live_root_size_bytes "${live_root_size_bytes}" \
    --arg low_margin_threshold_bytes "${low_margin_threshold_bytes}" \
    --arg min_free_bytes "${min_free_bytes}" \
    --arg observed_raw_span_days "${observed_raw_span_days}" \
    --arg estimated_growth_bytes_per_day "${estimated_growth_bytes_per_day}" \
    --arg status_days_to_min_free "${status_days_to_min_free}" \
    --arg estimated_days_to_ready "${estimated_days_to_ready}" \
    --arg status_min_free_before_ready "${status_min_free_before_ready}" \
    --arg bytes_to_clear_block "${bytes_to_clear_block}" \
    --arg bytes_to_clear_low_margin "${bytes_to_clear_low_margin}" \
    --arg live_root "${LIVE_ROOT}" \
    --arg current_fs_free_bytes "${current_fs_free_bytes}" \
    --arg current_fs_headroom_ok "${current_fs_headroom_ok}" \
    --arg current_fs_low_margin "${current_fs_low_margin}" \
    --arg current_fs_margin_bytes "${current_fs_margin_bytes}" \
    --arg current_reclaim_to_clear_block "${current_reclaim_to_clear_block}" \
    --arg current_reclaim_to_clear_low_margin "${current_reclaim_to_clear_low_margin}" \
    --arg current_days_to_min_free "${current_days_to_min_free}" \
    --arg current_min_free_before_ready "${current_min_free_before_ready}" \
    --arg status_level "${status_level}" \
    --argjson exit_code "${script_exit_code}" \
    --argjson reclaim_candidates "${candidate_inventory_json}" \
    --argjson live_roots "${live_roots_json}" \
    '
    def maybe_num($s):
      if $s == "" or $s == "n/a" or $s == "null" then null else ($s | tonumber? // null) end;
    def maybe_bool($s):
      if $s == "true" then true elif $s == "false" then false else null end;
    {
      status_path: $status_path,
      generated_at: (if $generated_at == "n/a" then null else $generated_at end),
      status_artifact: {
        generated_at: (if $generated_at == "n/a" then null else $generated_at end),
        age_seconds: maybe_num($status_age_seconds),
        max_age_seconds: maybe_num($status_max_age_seconds),
        fresh: maybe_bool($status_artifact_fresh)
      },
      status_level: $status_level,
      blocked: ($status_level == "blocked"),
      exit_code: $exit_code,
      screen: {
        session: (if $screen_session == "n/a" then null else $screen_session end),
        state: (if $screen_state == "n/a" then null else $screen_state end)
      },
      readiness: {
        ready_for_training: maybe_bool($ready_for_training),
        estimated_ready_at: (if $estimated_ready_at == "n/a" then null else $estimated_ready_at end),
        features_15m_v1: {
          target_progress_pct: maybe_num($feature_progress),
          remaining_target_days: maybe_num($feature_remaining_days),
          limiting_family: (if $feature_limiting_family == "n/a" then null else $feature_limiting_family end)
        },
        labels_15m_v1: {
          target_progress_pct: maybe_num($label_progress),
          remaining_target_days: maybe_num($label_remaining_days),
          limiting_family: (if $label_limiting_family == "n/a" then null else $label_limiting_family end)
        },
        quarantine_clean_window: {
          target_progress_pct: maybe_num($quarantine_progress),
          remaining_target_days: maybe_num($quarantine_remaining_days),
          estimated_ready_at: (if $quarantine_ready_at == "n/a" then null else $quarantine_ready_at end),
          quarantined_count: maybe_num($quarantined_count),
          latest_quarantined_path: (if $latest_quarantined_path == "n/a" then null else $latest_quarantined_path end)
        }
      },
      liveness: {
        raw_segments_fresh: maybe_bool($raw_fresh),
        processed_manifest_fresh: maybe_bool($manifest_fresh),
        latest_raw_age_seconds: maybe_num($latest_raw_age),
        latest_processed_age_seconds: maybe_num($latest_processed_age)
      },
      health: {
        invalid_recent_gzip_count: maybe_num($invalid_count),
        unrecovered_error_match_count: maybe_num($unrecovered_error_count)
      },
      disk_headroom: {
        source: "status_artifact",
        headroom_ok: maybe_bool($headroom_ok),
        headroom_low_margin: maybe_bool($headroom_low_margin),
        free_bytes: maybe_num($free_bytes),
        required_free_bytes: maybe_num($required_free_bytes),
        projected_remaining_bytes: maybe_num($projected_remaining_bytes),
        headroom_margin_bytes: maybe_num($headroom_margin_bytes),
        live_root_size_bytes: maybe_num($live_root_size_bytes),
        min_free_bytes: maybe_num($min_free_bytes),
        low_margin_threshold_bytes: maybe_num($low_margin_threshold_bytes),
        reclaim_to_clear_block_bytes: maybe_num($bytes_to_clear_block),
        reclaim_to_clear_low_margin_bytes: maybe_num($bytes_to_clear_low_margin)
      },
      current_filesystem_headroom: {
        live_root: $live_root,
        headroom_ok: maybe_bool($current_fs_headroom_ok),
        headroom_low_margin: maybe_bool($current_fs_low_margin),
        free_bytes: maybe_num($current_fs_free_bytes),
        required_free_bytes: maybe_num($required_free_bytes),
        projected_remaining_bytes: maybe_num($projected_remaining_bytes),
        headroom_margin_bytes: maybe_num($current_fs_margin_bytes),
        low_margin_threshold_bytes: maybe_num($low_margin_threshold_bytes),
        reclaim_to_clear_block_bytes: maybe_num($current_reclaim_to_clear_block),
        reclaim_to_clear_low_margin_bytes: maybe_num($current_reclaim_to_clear_low_margin),
        blocked: (maybe_bool($current_fs_headroom_ok) == false)
      },
      disk_urgency: {
        observed_raw_span_days: maybe_num($observed_raw_span_days),
        estimated_growth_bytes_per_day: maybe_num($estimated_growth_bytes_per_day),
        min_free_bytes: maybe_num($min_free_bytes),
        estimated_days_to_ready: maybe_num($estimated_days_to_ready),
        status_days_to_min_free: maybe_num($status_days_to_min_free),
        current_filesystem_days_to_min_free: maybe_num($current_days_to_min_free),
        status_min_free_before_ready: maybe_bool($status_min_free_before_ready),
        current_filesystem_min_free_before_ready: maybe_bool($current_min_free_before_ready)
      },
      reclaim_candidates: $reclaim_candidates,
      live_roots: $live_roots
    }'
  )"
  if [[ -n "${OUTPUT_PATH}" ]]; then
    write_json_output "${OUTPUT_PATH}" "${json_payload}"
  fi
  printf '%s\n' "${json_payload}"
  exit "${script_exit_code}"
fi

echo "xgboost-v4 collection risk check"
echo "status_path=${STATUS_PATH}"
echo "generated_at=${status_generated_at} age_seconds=${status_age_seconds} max_age_seconds=${STATUS_MAX_AGE_SECONDS} fresh=${status_artifact_fresh}"
echo "screen=${screen_session} state=${screen_state}"
echo
echo "readiness:"
echo "  ready_for_training=${ready_for_training} estimated_ready_at=${estimated_ready_at}"
echo "  features progress=${feature_progress}% remaining_days=${feature_remaining_days} limiting_family=${feature_limiting_family}"
echo "  labels progress=${label_progress}% remaining_days=${label_remaining_days} limiting_family=${label_limiting_family}"
echo "  quarantine count=${quarantined_count} progress=${quarantine_progress}% remaining_days=${quarantine_remaining_days} estimated_ready_at=${quarantine_ready_at}"
echo "  latest_quarantined=${latest_quarantined_path}"
echo
echo "liveness:"
echo "  raw_segments_fresh=${raw_fresh} processed_manifest_fresh=${manifest_fresh}"
echo "  latest_raw_age_seconds=${latest_raw_age} latest_processed_age_seconds=${latest_processed_age}"
echo "  invalid_recent_gzip_count=${invalid_count} unrecovered_error_match_count=${unrecovered_error_count}"
echo
echo "disk headroom:"
echo "  headroom_ok=${headroom_ok} headroom_low_margin=${headroom_low_margin}"
echo "  free=$(format_gib "${free_bytes}") required=$(format_gib "${required_free_bytes}") projected_remaining=$(format_gib "${projected_remaining_bytes}") margin=$(format_gib "${headroom_margin_bytes}")"
echo "  live_root_size=$(format_gib "${live_root_size_bytes}")"
echo "  reclaim_to_clear_block=$(format_gib "${bytes_to_clear_block}") reclaim_to_clear_low_margin=$(format_gib "${bytes_to_clear_low_margin}")"
echo "  current_fs_free=$(format_gib "${current_fs_free_bytes}") current_fs_headroom_ok=${current_fs_headroom_ok} current_fs_low_margin=${current_fs_low_margin} current_fs_margin=$(format_gib "${current_fs_margin_bytes}")"
echo "  current_reclaim_to_clear_block=$(format_gib "${current_reclaim_to_clear_block}") current_reclaim_to_clear_low_margin=$(format_gib "${current_reclaim_to_clear_low_margin}")"
echo "  urgency_estimate growth_per_day=$(format_gib "${estimated_growth_bytes_per_day}") min_free=$(format_gib "${min_free_bytes}") estimated_days_to_ready=${estimated_days_to_ready} status_days_to_min_free=${status_days_to_min_free} current_days_to_min_free=${current_days_to_min_free}"
echo "  min_free_before_ready status=${status_min_free_before_ready} current=${current_min_free_before_ready}"

if [[ -e "${LIVE_ROOT}" ]]; then
  echo
  echo "filesystem containing live root:"
  df -h "${LIVE_ROOT}"
fi

echo
echo "non-destructive reclaim-candidate inventory:"
for line in "${candidate_lines[@]}"; do
  echo "${line}"
done

if [[ "${SHOW_LIVE_ROOTS}" == "true" && -d data/live ]]; then
  echo
  echo "xgboost-v4 live roots:"
  for line in "${live_root_lines[@]}"; do
    echo "${line}"
  done
fi

echo
if [[ "${headroom_ok}" == "false" ]]; then
  if [[ "${status_artifact_fresh}" == "false" ]]; then
    echo "WARNING: status artifact is stale; regenerate live-collection-status before making final operator decisions."
  fi
  echo "ERROR: disk headroom is blocked. Free space is below projected requirement."
  echo "ACTION NEEDED (status artifact): reclaim at least $(format_gib "${bytes_to_clear_block}") to clear the hard block, or $(format_gib "${bytes_to_clear_low_margin}") to clear the low-margin buffer."
  if [[ "${current_reclaim_to_clear_block}" != "n/a" ]]; then
    echo "ACTION NEEDED (current filesystem): reclaim at least $(format_gib "${current_reclaim_to_clear_block}") to clear the hard block, or $(format_gib "${current_reclaim_to_clear_low_margin}") to clear the low-margin buffer."
  fi
  echo "This script is read-only; get explicit approval before pruning Docker, deleting simulator data, removing caches, stopping collection, or deleting old roots."
  exit "${script_exit_code}"
fi

if [[ "${headroom_low_margin}" == "true" ]]; then
  if [[ "${status_artifact_fresh}" == "false" ]]; then
    echo "WARNING: status artifact is stale; regenerate live-collection-status before making final operator decisions."
  fi
  echo "WARNING: disk headroom passes but margin is low. Review reclaim candidates before the collection fills the disk."
  echo "Suggested reclaim target to clear the low-margin buffer from status artifact: $(format_gib "${bytes_to_clear_low_margin}")."
  if [[ "${current_reclaim_to_clear_low_margin}" != "n/a" ]]; then
    echo "Suggested reclaim target to clear the low-margin buffer from current filesystem: $(format_gib "${current_reclaim_to_clear_low_margin}")."
  fi
else
  if [[ "${status_artifact_fresh}" == "false" ]]; then
    echo "WARNING: status artifact is stale; regenerate live-collection-status before making final operator decisions."
  fi
  echo "OK: disk headroom is not blocked."
fi
