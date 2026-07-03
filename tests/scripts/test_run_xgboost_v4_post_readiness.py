"""Contracts for the post-readiness xgboost-v4 shell runner."""

from __future__ import annotations

import json
import os
import shlex
import stat
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "run_xgboost_v4_post_readiness.sh"


def _write_fake_python(tmp_path: Path) -> Path:
    fake_python = tmp_path / "fake-python"
    real_python = shlex.quote(sys.executable)
    fake_python.write_text(
        f"""#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "${{1:-}}" == "-" || "${{1:-}}" == "-c" ]]; then
  exec {real_python} "$@"
fi

get_arg() {{
  local name="$1"
  shift
  while (($#)); do
    if [[ "$1" == "${{name}}" ]]; then
      echo "$2"
      return 0
    fi
    shift
  done
  return 1
}}

has_arg() {{
  local name="$1"
  shift
  while (($#)); do
    if [[ "$1" == "${{name}}" ]]; then
      return 0
    fi
    shift
  done
  return 1
}}

write_json() {{
  local path="$1"
  local payload="$2"
  mkdir -p "$(dirname "${{path}}")"
  printf '%s\\n' "${{payload}}" > "${{path}}"
}}

write_dir_artifacts() {{
  local output_dir="$1"
  mkdir -p "${{output_dir}}"
  write_json "${{output_dir}}/manifest.json" '{{"ok": true}}'
}}

if [[ "${{1:-}}" == "-m" && "${{2:-}}" == "bigan.ingestion.__main__" ]]; then
  command="${{3:-}}"
  shift 3
  if [[ -n "${{FAKE_COMMAND_LOG:-}}" ]]; then
    printf '%s' "${{command}}" >> "${{FAKE_COMMAND_LOG}}"
    for arg in "$@"; do
      printf '\\t%s' "${{arg}}" >> "${{FAKE_COMMAND_LOG}}"
    done
    printf '\\n' >> "${{FAKE_COMMAND_LOG}}"
  fi
  case "${{command}}" in
    live-collection-readiness)
      if [[ "${{FAKE_READY:-false}}" == "true" ]]; then
        printf '%s\\n' '{{"ready": true, "ready_for_training": true, "estimated_ready_at": null}}'
      else
        printf '%s\\n' '{{"ready": false, "ready_for_training": false, "blockers": ["not ready"], "estimated_ready_at": "soon"}}'
      fi
      ;;
    labels-15m-v1)
      printf '%s\\n' '{{"rows_written": 0}}'
      ;;
    live-collection-status)
      output_path="$(get_arg --output-path "$@")"
      write_json "${{output_path}}" "$(printf '{{"generated_at": "2026-05-30T12:00:00Z", "live_root": "%s", "warehouse": "%s/warehouse", "screen_session": "%s", "screen_state": "running", "raw_segment_count": 10, "processed_manifest_rows": 9, "raw_segment_quarantine": {{"quarantined_count": 0, "latest_quarantined_segment": null}}, "collection_readiness": {{"ready_for_training": true, "estimated_ready_at": null, "required_families": ["BTC-15M", "ETH-15M", "BTC-5M", "ETH-5M"], "quarantine_clean_window": {{"meets_target": true, "target_progress_pct": 100.0, "remaining_target_days": 0.0, "estimated_ready_at": null}}, "features_15m_v1": {{"target_progress_pct": 100.0, "remaining_target_days": 0.0, "limiting_family": "ETH-15M"}}, "labels_15m_v1": {{"target_progress_pct": 100.0, "remaining_target_days": 0.0, "limiting_family": "BTC-15M"}}}}, "family_spans": {{"features_15m_v1": {{"BTC-15M": {{"min_ts": 1779600000000, "max_ts": 1779686400000}}, "ETH-15M": {{"min_ts": 1779600000000, "max_ts": 1779686400000}}, "BTC-5M": {{"min_ts": 1779600000000, "max_ts": 1779686400000}}, "ETH-5M": {{"min_ts": 1779600000000, "max_ts": 1779686400000}}}}}}, "disk_headroom_evidence": {{"path": "%s", "headroom_ok": %s, "headroom_low_margin": false, "free_bytes": %s, "required_free_bytes": %s, "headroom_margin_bytes": %s, "low_margin_threshold_bytes": %s}}, "health_evidence": {{"unrecovered_error_match_count": 0}}}}' "${{LIVE_ROOT}}" "${{LIVE_ROOT}}" "${{SCREEN_SESSION}}" "${{LIVE_ROOT}}" "${{FAKE_HEADROOM_OK:-true}}" "${{FAKE_FREE_BYTES:-107374182400}}" "${{FAKE_REQUIRED_FREE_BYTES:-1073741824}}" "${{FAKE_MARGIN_BYTES:-106300440576}}" "${{FAKE_LOW_MARGIN_THRESHOLD_BYTES:-1073741824}}")"
      ;;
    champion-state-snapshot-v1)
      output_path="$(get_arg --output-path "$@")"
      write_json "${{output_path}}" "$(printf '{{"registry_champion": {{"artifact_uri": "%s", "model_version": "xgboost-v3", "calibration_artifact_uri": "%s"}}, "online_model": {{"rollback_to_version": "xgboost-v2"}}, "fallback_registry_model": {{"artifact_uri": "%s"}}}}' "${{FAKE_INCUMBENT_MODEL}}" "${{FAKE_INCUMBENT_CALIBRATION}}" "${{FAKE_FALLBACK_MODEL}}")"
      ;;
    training-dataset-v1|model-eval-v1)
      output_dir="$(get_arg --output-dir "$@")"
      write_dir_artifacts "${{output_dir}}"
      write_json "${{output_dir}}/offline_reference.json" '{{"model_version": "xgboost-v4"}}'
      ;;
    dataset-stability-report-v1)
      output_dir="$(get_arg --output-dir "$@")"
      write_dir_artifacts "${{output_dir}}"
      write_json "${{output_dir}}/dataset_stability_report.json" '{{"schema_version": "dataset_stability_report_v1"}}'
      printf '%s\n' '# Dataset Stability Report' > "${{output_dir}}/dataset_stability_report.md"
      ;;
    xgboost-v4)
      output_dir="$(get_arg --output-dir "$@")"
      write_dir_artifacts "${{output_dir}}"
      write_json "${{output_dir}}/model.json" '{{"model_version": "xgboost-v4"}}'
      write_json "${{output_dir}}/feature_schema.json" '{{"features": []}}'
      ;;
    calibration-v1)
      output_dir="$(get_arg --output-dir "$@")"
      write_dir_artifacts "${{output_dir}}"
      write_json "${{output_dir}}/calibration.json" '{{"calibration_method": "identity"}}'
      ;;
    offline-rerun-report-v1)
      output_path="$(get_arg --output-path "$@")"
      mkdir -p "$(dirname "${{output_path}}")"
      printf '%s\\n' '# Rerun Report' > "${{output_path}}"
      write_json "${{output_path%.md}}.json" '{{"passed": true}}'
      ;;
    feature-ablation-report-v1)
      output_dir="$(get_arg --output-dir "$@")"
      write_dir_artifacts "${{output_dir}}"
      write_json "${{output_dir}}/feature_ablation.json" '{{"passed": true}}'
      ;;
    backtest-model-v1)
      output_dir="$(get_arg --output-dir "$@")"
      write_dir_artifacts "${{output_dir}}"
      write_json "${{output_dir}}/summary.json" '{{"summary": []}}'
      write_json "${{output_dir}}/diagnostics.json" '{{"summary": []}}'
      ;;
    serving-readiness-v1)
      output_path="$(get_arg --output-path "$@")"
      write_json "${{output_path}}" '{{"ready": true}}'
      ;;
    shadow-v1)
      output_path="$(get_arg --output-path "$@")"
      evaluation_json_path="$(get_arg --evaluation-json-output-path "$@")"
      since_ms="$(get_arg --since-ms "$@")"
      until_ms="$(get_arg --until-ms "$@")"
      offline_reference_path="$(get_arg --offline-reference-path "$@")"
      session_duration_seconds="$(( (until_ms - since_ms) / 1000 ))"
      shadow_payload="$(printf '{{"overall_passed": true, "champion_model_version": "xgboost-v3", "challenger_model_version": "xgboost-v4", "sample_count": 10, "scored_count": 10, "challenger_probability_distribution": {{"count": 10, "mean": 0.56, "std": 0.105}}, "schema_error_rate": 0.0, "scoring_error_rate": 0.0, "offline_reference_path": "%s", "window_start_ts": %s, "window_end_ts": %s, "session_duration_seconds": %s}}' "${{offline_reference_path}}" "${{since_ms}}" "${{until_ms}}" "${{session_duration_seconds}}")"
      write_json "${{output_path}}" "${{shadow_payload}}"
      write_json "${{evaluation_json_path}}" "${{shadow_payload}}"
      ;;
    bootstrap-champion-v1)
      output_dir="$(get_arg --output-dir "$@")"
      write_dir_artifacts "${{output_dir}}"
      write_json "${{output_dir}}/bootstrap_decision.json" '{{"recommended_action": "PROMOTE_CHAMPION"}}'
      ;;
    drift-baseline-v1)
      output_path="$(get_arg --output-path "$@")"
      write_json "${{output_path}}" '{{"model_version": "xgboost-v4"}}'
      ;;
    champion-cutover-report-v1)
      output_path="$(get_arg --output-path "$@")"
      write_json "${{output_path}}" '{{"passed": false}}'
      ;;
    champion-promotion-audit)
      output_dir="$(get_arg --output-dir "$@")"
      write_dir_artifacts "${{output_dir}}"
      if [[ "${{FAKE_PROMOTION_PASSED:-false}}" == "true" ]]; then
        write_json "${{output_dir}}/champion_promotion_audit.json" '{{"passed": true, "decision": "PROMOTION_COMPLETE"}}'
      else
        write_json "${{output_dir}}/champion_promotion_audit.json" '{{"passed": false, "decision": "BLOCKED"}}'
      fi
      if [[ "${{FAKE_PROMOTION_PASSED:-false}}" != "true" ]] && ! has_arg --no-fail-on-blocked "$@"; then
        exit 1
      fi
      ;;
    xgboost-v4-objective-audit)
      output_path="$(get_arg --output-path "$@")"
      latest_path="$(get_arg --post-readiness-latest-path "$@" || true)"
      if [[ "${{FAKE_OBJECTIVE_COMPLETES_AFTER_POINTER:-false}}" == "true" && -n "${{latest_path}}" && -f "${{latest_path}}" ]]; then
        write_json "${{output_path}}" '{{"objective_complete": true, "decision": "COMPLETE", "objective_restatement": {{"summary": "Create xgboost-v4", "slack_channel_id": "C0B5VHYSCN8"}}, "objective_success_criteria": [{{"id": "all_requested_github_issues_satisfied", "passed": true}}, {{"id": "fresh_xgboost_v4_model_created", "passed": true}}, {{"id": "beats_current_champion", "passed": true}}, {{"id": "champion_promotion_gates_passed", "passed": true}}, {{"id": "hourly_slack_status_active", "passed": true}}, {{"id": "post_readiness_latest_pointer_valid", "passed": true}}], "blockers": [], "prompt_to_artifact_blockers": [], "prompt_to_artifact_checklist": [{{"id": "github_issue_54", "passed": true}}, {{"id": "github_issue_55", "passed": true}}, {{"id": "github_issue_56", "passed": true}}, {{"id": "github_issue_57", "passed": true}}, {{"id": "github_issue_58", "passed": true}}, {{"id": "github_issue_64", "passed": true}}, {{"id": "github_issue_65", "passed": true}}, {{"id": "create_xgboost_v4_model", "passed": true}}, {{"id": "beat_current_champion", "passed": true}}, {{"id": "champion_promotion_md", "passed": true}}, {{"id": "hourly_slack_status", "passed": true}}, {{"id": "post_readiness_latest_pointer", "passed": true}}], "promotion": {{"raw_passed": true, "clean_atomic_live_root_passed": true, "status_artifact_fresh_passed": true, "passed": true}}}}'
        exit 0
      elif [[ "${{FAKE_OBJECTIVE_ONLY_POINTER_BLOCKED:-false}}" == "true" ]]; then
        write_json "${{output_path}}" '{{"objective_complete": false, "decision": "BLOCKED", "objective_restatement": {{"summary": "Create xgboost-v4", "slack_channel_id": "C0B5VHYSCN8"}}, "objective_success_criteria": [{{"id": "post_readiness_latest_pointer_valid", "passed": false}}], "blockers": ["post_readiness_latest_pointer: latest pointer missing"], "prompt_to_artifact_blockers": ["post_readiness_latest_pointer: latest pointer missing"], "prompt_to_artifact_checklist": [{{"id": "post_readiness_latest_pointer", "passed": false}}], "promotion": {{"raw_passed": true, "clean_atomic_live_root_passed": true, "status_artifact_fresh_passed": true, "passed": true}}}}'
      else
        write_json "${{output_path}}" '{{"objective_complete": false, "decision": "BLOCKED", "objective_restatement": {{"summary": "Create xgboost-v4", "slack_channel_id": "C0B5VHYSCN8"}}, "objective_success_criteria": [{{"id": "fresh_xgboost_v4_model_created", "passed": false}}, {{"id": "hourly_slack_status_active", "passed": true}}], "blockers": ["shadow evidence missing", "create_xgboost_v4_model: model not final"], "prompt_to_artifact_blockers": ["create_xgboost_v4_model: model not final"], "prompt_to_artifact_checklist": [{{"id": "create_xgboost_v4_model", "passed": false}}, {{"id": "hourly_slack_status", "passed": true}}], "promotion": {{"raw_passed": false, "clean_atomic_live_root_passed": true, "status_artifact_fresh_passed": true, "passed": false}}}}'
      fi
      if ! has_arg --no-fail-on-blocked "$@"; then
        exit 1
      fi
      ;;
    xgboost-v4-issue-coverage-audit)
      output_path="$(get_arg --output-path "$@")"
      if [[ "${{FAKE_ISSUE_COVERAGE_PASSED:-false}}" == "true" ]]; then
        write_json "${{output_path}}" '{{"generated_at": "2026-05-30T12:01:00+00:00", "summary": {{"decision": "COMPLETE", "objective_complete": true, "blocker_count": 0}}, "issue_checks": {{"#54": {{"passed": true}}, "#55": {{"passed": true}}, "#56": {{"passed": true}}, "#57": {{"passed": true}}, "#58": {{"passed": true}}, "#64": {{"passed": true}}, "#65": {{"passed": true}}}}, "objective_success_criteria": {{"all_requested_github_issues_satisfied": {{"passed": true}}, "fresh_xgboost_v4_model_created": {{"passed": true}}, "beats_current_champion": {{"passed": true}}, "champion_promotion_gates_passed": {{"passed": true}}, "hourly_slack_status_active": {{"passed": true}}, "post_readiness_latest_pointer_valid": {{"passed": true}}}}}}'
      else
        write_json "${{output_path}}" '{{"generated_at": "2026-05-30T12:01:00+00:00", "summary": {{"decision": "BLOCKED", "objective_complete": false, "blocker_count": 2}}, "issue_checks": {{"#54": {{"passed": true}}, "#55": {{"passed": false}}}}, "objective_success_criteria": {{"fresh_xgboost_v4_model_created": {{"passed": false}}, "hourly_slack_status_active": {{"passed": true}}}}}}'
      fi
      ;;
    *)
      printf 'unexpected command: %s\\n' "${{command}}" >&2
      exit 12
      ;;
  esac
  exit 0
fi

exec {real_python} "$@"
""",
        encoding="utf-8",
    )
    fake_python.chmod(fake_python.stat().st_mode | stat.S_IXUSR)
    return fake_python


def _runner_env(
    tmp_path: Path,
    *,
    ready: bool,
    run_root_name: str = "run",
) -> tuple[dict[str, str], Path]:
    fake_python = _write_fake_python(tmp_path)
    status_path = tmp_path / "status.json"
    manifest_path = tmp_path / "processed.txt"
    live_root = tmp_path / "xgboost-v4-multimarket-7d-atomic-live-root"
    live_root.mkdir(parents=True, exist_ok=True)
    clean_window = {
        "meets_target": True,
        "target_progress_pct": 100.0,
        "remaining_target_days": 0.0,
        "estimated_ready_at": None,
    }
    status_path.write_text(
        json.dumps(
            {
                "generated_at": "2026-05-30T12:00:00Z",
                "live_root": str(live_root),
                "warehouse": str(live_root / "warehouse"),
                "screen_session": "test_screen",
                "screen_state": "running",
                "raw_segment_count": 10,
                "processed_manifest_rows": 9,
                "raw_segment_quarantine": {
                    "quarantined_count": 0,
                    "latest_quarantined_segment": None,
                },
                "collection_readiness": {
                    "ready_for_training": ready,
                    "estimated_ready_at": None if ready else "soon",
                    "required_families": ["BTC-15M", "ETH-15M", "BTC-5M", "ETH-5M"],
                    "quarantine_clean_window": clean_window,
                    "features_15m_v1": {
                        "target_progress_pct": 100.0 if ready else 50.0,
                        "remaining_target_days": 0.0 if ready else 3.5,
                        "limiting_family": "ETH-15M",
                    },
                    "labels_15m_v1": {
                        "target_progress_pct": 100.0 if ready else 45.0,
                        "remaining_target_days": 0.0 if ready else 3.85,
                        "limiting_family": "BTC-15M",
                    },
                },
                "family_spans": {
                    "features_15m_v1": {
                        "BTC-15M": {"min_ts": 1779600000000, "max_ts": 1779686400000},
                        "ETH-15M": {"min_ts": 1779600000000, "max_ts": 1779686400000},
                        "BTC-5M": {"min_ts": 1779600000000, "max_ts": 1779686400000},
                        "ETH-5M": {"min_ts": 1779600000000, "max_ts": 1779686400000},
                    }
                },
                "health_evidence": {"unrecovered_error_match_count": 0},
                "disk_headroom_evidence": {
                    "headroom_ok": True,
                    "headroom_low_margin": False,
                    "free_bytes": 100 * 1024**3,
                    "required_free_bytes": 1 * 1024**3,
                    "headroom_margin_bytes": 99 * 1024**3,
                },
            }
        ),
        encoding="utf-8",
    )
    manifest_path.write_text("", encoding="utf-8")

    incumbent_model = tmp_path / "incumbent-model.json"
    incumbent_calibration = tmp_path / "incumbent-calibration.json"
    fallback_model = tmp_path / "fallback-model.json"
    for path in (incumbent_model, incumbent_calibration, fallback_model):
        path.write_text("{}", encoding="utf-8")

    run_root = tmp_path / run_root_name
    env = {
        **os.environ,
        "PYTHON_BIN": str(fake_python),
        "STATUS_PATH": str(status_path),
        "MANIFEST_PATH": str(manifest_path),
        "LIVE_ROOT": str(live_root),
        "SCREEN_SESSION": "test_screen",
        "MONITORING_DB_PATH": str(tmp_path / "mlops.duckdb"),
        "RUN_ROOT": str(run_root),
        "RUN_ID": "test-run",
        "POST_READINESS_SENTINEL_PATH": str(tmp_path / "post-readiness-completed.json"),
        "POST_READINESS_LOCK_DIR": str(tmp_path / "post-readiness.lock"),
        "POST_READINESS_LATEST_PATH": str(tmp_path / "post-readiness-latest.json"),
        "SLACK_DELIVERY_STATUS_PATH": str(tmp_path / "slack-delivery-status.json"),
        "FAKE_READY": "true" if ready else "false",
        "FAKE_INCUMBENT_MODEL": str(incumbent_model),
        "FAKE_INCUMBENT_CALIBRATION": str(incumbent_calibration),
        "FAKE_FALLBACK_MODEL": str(fallback_model),
    }
    return env, run_root


def _run_script(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/bin/bash", str(SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )


def test_plan_only_prints_paths_without_creating_run_root(tmp_path: Path) -> None:
    env, run_root = _runner_env(tmp_path, ready=False)
    env["PLAN_ONLY"] = "true"

    result = _run_script(env)

    assert result.returncode == 0
    assert "PLAN_ONLY=true" in result.stdout
    assert "run_manifest=" in result.stdout
    assert f"slack_delivery_status={env['SLACK_DELIVERY_STATUS_PATH']}" in result.stdout
    assert "issue_coverage_audit=" in result.stdout
    assert "not ready" in result.stdout
    assert not run_root.exists()


def test_not_ready_default_aborts_before_creating_run_root(tmp_path: Path) -> None:
    env, run_root = _runner_env(tmp_path, ready=False)
    command_log = tmp_path / "commands.log"
    env["FAKE_COMMAND_LOG"] = str(command_log)

    result = _run_script(env)

    assert result.returncode == 1
    assert "corpus is not ready" in result.stderr
    assert not run_root.exists()
    lines = command_log.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("live-collection-status\t")
    assert lines[1].startswith("live-collection-readiness\t")


def test_plan_only_uses_existing_status_without_refreshing(tmp_path: Path) -> None:
    env, run_root = _runner_env(tmp_path, ready=False)
    command_log = tmp_path / "commands.log"
    env["PLAN_ONLY"] = "true"
    env["FAKE_COMMAND_LOG"] = str(command_log)

    result = _run_script(env)

    assert result.returncode == 0
    lines = command_log.read_text(encoding="utf-8").splitlines()
    assert lines == [
        f"live-collection-readiness\t--status-path\t{env['STATUS_PATH']}\t--no-fail-on-blocked"
    ]
    assert not run_root.exists()


def test_plan_only_reports_blocked_disk_headroom_without_creating_run_root(
    tmp_path: Path,
) -> None:
    env, run_root = _runner_env(tmp_path, ready=True)
    command_log = tmp_path / "commands.log"
    env["PLAN_ONLY"] = "true"
    env["FAKE_COMMAND_LOG"] = str(command_log)
    status_path = Path(env["STATUS_PATH"])
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    payload["disk_headroom_evidence"].update(
        {
            "headroom_ok": False,
            "free_bytes": 32 * 1024**3,
            "required_free_bytes": 44 * 1024**3,
            "headroom_margin_bytes": -12 * 1024**3,
        }
    )
    status_path.write_text(json.dumps(payload), encoding="utf-8")

    result = _run_script(env)

    assert result.returncode == 0
    assert "disk headroom is blocked" in result.stderr
    assert "PLAN_ONLY continuing after disk headroom warning" in result.stderr
    assert "PLAN_ONLY=true" in result.stdout
    assert "current readiness" in result.stdout
    assert not run_root.exists()
    lines = command_log.read_text(encoding="utf-8").splitlines()
    assert lines == [
        f"live-collection-readiness\t--status-path\t{env['STATUS_PATH']}\t--no-fail-on-blocked"
    ]


def test_status_artifact_must_match_clean_live_root_before_readiness(tmp_path: Path) -> None:
    env, run_root = _runner_env(tmp_path, ready=True)
    status_path = Path(env["STATUS_PATH"])
    status_path.write_text(
        json.dumps(
            {
                "live_root": str(tmp_path / "pre-atomic-debug-root"),
                "warehouse": str(tmp_path / "pre-atomic-debug-root" / "warehouse"),
                "screen_session": env["SCREEN_SESSION"],
            }
        ),
        encoding="utf-8",
    )
    env["PLAN_ONLY"] = "true"

    result = _run_script(env)

    assert result.returncode == 1
    assert "does not describe the configured clean corpus" in result.stderr
    assert "pre-atomic-debug-root" in result.stderr
    assert "current readiness" not in result.stdout
    assert not run_root.exists()


def test_ready_pipeline_refreshes_stale_status_before_matching_live_root(
    tmp_path: Path,
) -> None:
    env, run_root = _runner_env(tmp_path, ready=True)
    status_path = Path(env["STATUS_PATH"])
    status_path.write_text(
        json.dumps(
            {
                "live_root": str(tmp_path / "pre-atomic-debug-root"),
                "warehouse": str(tmp_path / "pre-atomic-debug-root" / "warehouse"),
                "screen_session": env["SCREEN_SESSION"],
            }
        ),
        encoding="utf-8",
    )
    command_log = tmp_path / "commands.log"
    env["FAKE_COMMAND_LOG"] = str(command_log)

    result = _run_script(env)

    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["live_root"] == env["LIVE_ROOT"]
    lines = command_log.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("live-collection-status\t")
    assert lines[1].startswith("live-collection-readiness\t")
    assert (run_root / "artifacts" / "run_manifest.json").exists()


def test_non_atomic_live_root_is_rejected_before_status_refresh(tmp_path: Path) -> None:
    env, run_root = _runner_env(tmp_path, ready=True)
    command_log = tmp_path / "commands.log"
    env["FAKE_COMMAND_LOG"] = str(command_log)
    env["LIVE_ROOT"] = str(tmp_path / "pre-atomic-debug-root")

    result = _run_script(env)

    assert result.returncode == 1
    assert "LIVE_ROOT must point at the clean xgboost-v4 multimarket 7d atomic corpus root" in result.stderr
    assert not command_log.exists()
    assert not run_root.exists()


def test_shadow_mode_requires_explicit_bounds_before_readiness(tmp_path: Path) -> None:
    env, _ = _runner_env(tmp_path, ready=False)
    env["PLAN_ONLY"] = "true"
    env["RUN_SHADOW"] = "true"

    result = _run_script(env)

    assert result.returncode == 1
    assert "RUN_SHADOW=true requires SHADOW_SINCE_MS and SHADOW_UNTIL_MS" in result.stderr
    assert "current readiness" not in result.stdout


def test_shadow_mode_rejects_non_integer_bounds_before_readiness(tmp_path: Path) -> None:
    env, _ = _runner_env(tmp_path, ready=False)
    env["PLAN_ONLY"] = "true"
    env["RUN_SHADOW"] = "true"
    env["SHADOW_SINCE_MS"] = "soon"
    env["SHADOW_UNTIL_MS"] = "1779686400000"

    result = _run_script(env)

    assert result.returncode == 1
    assert "SHADOW_SINCE_MS and SHADOW_UNTIL_MS must be integer epoch milliseconds" in result.stderr
    assert "current readiness" not in result.stdout


def test_shadow_mode_rejects_one_sided_auto_bounds_before_readiness(tmp_path: Path) -> None:
    env, _ = _runner_env(tmp_path, ready=False)
    env["PLAN_ONLY"] = "true"
    env["RUN_SHADOW"] = "true"
    env["SHADOW_SINCE_MS"] = "auto"
    env["SHADOW_UNTIL_MS"] = "1779686400000"

    result = _run_script(env)

    assert result.returncode == 1
    assert "use auto for both SHADOW_SINCE_MS and SHADOW_UNTIL_MS" in result.stderr
    assert "current readiness" not in result.stdout


def test_shadow_mode_rejects_short_window_before_readiness(tmp_path: Path) -> None:
    env, _ = _runner_env(tmp_path, ready=False)
    env["PLAN_ONLY"] = "true"
    env["RUN_SHADOW"] = "true"
    env["SHADOW_SINCE_MS"] = "1779600000000"
    env["SHADOW_UNTIL_MS"] = "1779600000001"

    result = _run_script(env)

    assert result.returncode == 1
    assert "shadow window is too short" in result.stderr
    assert "required_ms>=86400000" in result.stderr
    assert "current readiness" not in result.stdout


def test_shadow_mode_plan_only_resolves_auto_bounds_from_status(tmp_path: Path) -> None:
    env, _ = _runner_env(tmp_path, ready=True)
    env["PLAN_ONLY"] = "true"
    env["RUN_SHADOW"] = "true"
    env["SHADOW_SINCE_MS"] = "auto"
    env["SHADOW_UNTIL_MS"] = "auto"

    result = _run_script(env)

    assert result.returncode == 0, result.stderr + result.stdout
    assert "auto shadow window resolved" in result.stdout
    assert "from 1779600000000 to 1779686400000" in result.stdout


def test_shadow_mode_rejects_auto_bounds_when_common_span_is_too_short(tmp_path: Path) -> None:
    env, _ = _runner_env(tmp_path, ready=True)
    env["PLAN_ONLY"] = "true"
    env["RUN_SHADOW"] = "true"
    env["SHADOW_SINCE_MS"] = "auto"
    env["SHADOW_UNTIL_MS"] = "auto"
    status_path = Path(env["STATUS_PATH"])
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    family_spans = payload["family_spans"]["features_15m_v1"]
    for span in family_spans.values():
        span["min_ts"] = 1779686399000
        span["max_ts"] = 1779686400000
    status_path.write_text(json.dumps(payload), encoding="utf-8")

    result = _run_script(env)

    assert result.returncode == 1
    assert "cannot resolve auto shadow window" in result.stderr
    assert "below required 86400000ms" in result.stderr


def test_shadow_generation_rejects_unwritable_custom_bootstrap_path(tmp_path: Path) -> None:
    env, _ = _runner_env(tmp_path, ready=False)
    env["PLAN_ONLY"] = "true"
    env["RUN_SHADOW"] = "true"
    env["SHADOW_SINCE_MS"] = "1779600000000"
    env["SHADOW_UNTIL_MS"] = "1779686400000"
    env["BOOTSTRAP_DECISION_PATH"] = str(tmp_path / "existing" / "bootstrap_decision.json")

    result = _run_script(env)

    assert result.returncode == 1
    assert "RUN_SHADOW=true writes bootstrap output" in result.stderr
    assert "current readiness" not in result.stdout


def test_shadow_generation_rejects_custom_bootstrap_path_with_auto_window(
    tmp_path: Path,
) -> None:
    env, _ = _runner_env(tmp_path, ready=False)
    env["PLAN_ONLY"] = "true"
    env["RUN_SHADOW"] = "true"
    env["SHADOW_SINCE_MS"] = "auto"
    env["SHADOW_UNTIL_MS"] = "auto"
    env["BOOTSTRAP_DECISION_PATH"] = str(tmp_path / "existing" / "bootstrap_decision.json")

    result = _run_script(env)

    assert result.returncode == 1
    assert "RUN_SHADOW=true writes bootstrap output" in result.stderr
    assert "current readiness" not in result.stdout


def test_cutover_mode_requires_inputs_with_auto_shadow_window(tmp_path: Path) -> None:
    env, _ = _runner_env(tmp_path, ready=False)
    env["PLAN_ONLY"] = "true"
    env["RUN_SHADOW"] = "true"
    env["SHADOW_SINCE_MS"] = "auto"
    env["SHADOW_UNTIL_MS"] = "auto"
    env["RUN_CUTOVER_REPORT"] = "true"

    result = _run_script(env)

    assert result.returncode == 1
    assert "missing cutover smoke artifact" in result.stderr
    assert "current readiness" not in result.stdout


def test_invalid_stale_lock_seconds_fails_before_readiness(tmp_path: Path) -> None:
    env, _ = _runner_env(tmp_path, ready=False)
    env["PLAN_ONLY"] = "true"
    env["POST_READINESS_LOCK_STALE_SECONDS"] = "soon"

    result = _run_script(env)

    assert result.returncode == 1
    assert "POST_READINESS_LOCK_STALE_SECONDS must be a non-negative integer" in result.stderr
    assert "current readiness" not in result.stdout


def test_ready_pipeline_writes_run_manifest(tmp_path: Path) -> None:
    env, run_root = _runner_env(tmp_path, ready=True)

    result = _run_script(env)

    assert result.returncode == 0, result.stderr + result.stdout
    manifest_path = run_root / "artifacts" / "run_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["phase"] == "completed"
    assert payload["readiness"]["ready"] is True
    assert payload["live_status_summary"]["exists"] is True
    assert payload["live_status_summary"]["live_root"] == env["LIVE_ROOT"]
    assert payload["live_status_summary"]["ready_for_training"] is True
    assert payload["live_status_summary"]["quarantined_raw_segments"] == 0
    assert payload["live_status_summary"]["quarantine_clean_window_ready"] is True
    assert payload["live_status_summary"]["feature_limiting_family"] == "ETH-15M"
    assert payload["live_status_summary"]["label_limiting_family"] == "BTC-15M"
    assert payload["settings"]["run_shadow"] is False
    assert payload["paths"]["candidate_model_path"].endswith("/artifacts/models/xgboost-v4/model.json")
    assert payload["paths"]["promotion_audit_path"].endswith(
        "/artifacts/champion-promotion-audit/champion_promotion_audit.json"
    )
    assert payload["paths"]["issue_coverage_audit_path"].endswith(
        "/artifacts/issue_coverage_audit.json"
    )
    assert payload["audit_results"]["promotion"]["decision"] == "BLOCKED"
    assert payload["audit_results"]["promotion"]["passed"] is False
    assert payload["audit_results"]["objective"]["decision"] == "BLOCKED"
    assert payload["audit_results"]["objective"]["objective_complete"] is False
    assert payload["audit_results"]["objective"]["restatement"] == {
        "summary": "Create xgboost-v4",
        "slack_channel_id": "C0B5VHYSCN8",
    }
    assert payload["audit_results"]["objective"]["success_criteria"] == [
        {"id": "fresh_xgboost_v4_model_created", "passed": False},
        {"id": "hourly_slack_status_active", "passed": True},
    ]
    assert payload["audit_results"]["objective"]["blockers"] == [
        "shadow evidence missing",
        "create_xgboost_v4_model: model not final",
    ]
    assert payload["audit_results"]["objective"]["prompt_to_artifact_blockers"] == [
        "create_xgboost_v4_model: model not final"
    ]
    assert payload["audit_results"]["objective"]["prompt_to_artifact_checklist"] == [
        {"id": "create_xgboost_v4_model", "passed": False},
        {"id": "hourly_slack_status", "passed": True},
    ]
    assert payload["audit_results"]["objective"]["promotion"] == {
        "raw_passed": False,
        "clean_atomic_live_root_passed": True,
        "status_artifact_fresh_passed": True,
        "passed": False,
    }
    assert payload["audit_results"]["issue_coverage"]["exists"] is True
    assert payload["audit_results"]["issue_coverage"]["decision"] == "BLOCKED"
    assert payload["audit_results"]["issue_coverage"]["objective_complete"] is False
    assert payload["audit_results"]["issue_coverage"]["blocker_count"] == 2
    assert payload["audit_results"]["issue_coverage"]["issue_checks"] == {
        "#54": {"passed": True},
        "#55": {"passed": False},
    }
    assert (run_root / "artifacts" / "xgboost_v4_objective_audit.json").exists()
    assert (run_root / "artifacts" / "issue_coverage_audit.json").exists()
    sentinel = json.loads(Path(env["POST_READINESS_SENTINEL_PATH"]).read_text(encoding="utf-8"))
    assert sentinel["completion_scope"] == "post_readiness_runner_completed"
    assert "objective may remain blocked" in sentinel["completion_note"]
    assert sentinel["run_root"] == str(run_root)
    assert sentinel["run_manifest_path"] == str(manifest_path)
    assert sentinel["run_manifest_phase"] == "completed"
    assert sentinel["live_status_summary"]["ready_for_training"] is True
    assert sentinel["live_status_summary"]["quarantine_clean_window_ready"] is True
    assert sentinel["artifact_paths"]["objective_audit_path"] == str(
        run_root / "artifacts" / "xgboost_v4_objective_audit.json"
    )
    assert sentinel["artifact_paths"]["promotion_audit_path"] == str(
        run_root / "artifacts" / "champion-promotion-audit" / "champion_promotion_audit.json"
    )
    assert sentinel["artifact_paths"]["issue_coverage_audit_path"] == str(
        run_root / "artifacts" / "issue_coverage_audit.json"
    )
    assert sentinel["promotion_decision"] == "BLOCKED"
    assert sentinel["promotion_passed"] is False
    assert sentinel["objective_decision"] == "BLOCKED"
    assert sentinel["objective_complete"] is False
    assert sentinel["objective_restatement"] == {
        "summary": "Create xgboost-v4",
        "slack_channel_id": "C0B5VHYSCN8",
    }
    assert sentinel["objective_success_criteria"] == [
        {"id": "fresh_xgboost_v4_model_created", "passed": False},
        {"id": "hourly_slack_status_active", "passed": True},
    ]
    assert sentinel["objective_blockers"] == [
        "shadow evidence missing",
        "create_xgboost_v4_model: model not final",
    ]
    assert sentinel["objective_prompt_to_artifact_blockers"] == [
        "create_xgboost_v4_model: model not final"
    ]
    assert sentinel["objective_prompt_to_artifact_checklist"] == [
        {"id": "create_xgboost_v4_model", "passed": False},
        {"id": "hourly_slack_status", "passed": True},
    ]
    assert sentinel["objective_promotion"] == {
        "raw_passed": False,
        "clean_atomic_live_root_passed": True,
        "status_artifact_fresh_passed": True,
        "passed": False,
    }
    assert sentinel["issue_coverage_audit_path"] == str(
        run_root / "artifacts" / "issue_coverage_audit.json"
    )
    assert sentinel["issue_coverage_generated_at"] == "2026-05-30T12:01:00+00:00"
    assert sentinel["issue_coverage_issue_checks"] == {
        "#54": {"passed": True},
        "#55": {"passed": False},
    }
    assert sentinel["sentinel_path"] == env["POST_READINESS_SENTINEL_PATH"]
    latest = json.loads(Path(env["POST_READINESS_LATEST_PATH"]).read_text(encoding="utf-8"))
    assert latest == sentinel
    assert not Path(env["POST_READINESS_LOCK_DIR"]).exists()
    assert not list(tmp_path.rglob("*.tmp"))


def test_ready_pipeline_refuses_blocked_disk_headroom_before_retrain(tmp_path: Path) -> None:
    env, run_root = _runner_env(tmp_path, ready=True)
    command_log = tmp_path / "commands.log"
    env["FAKE_COMMAND_LOG"] = str(command_log)
    env["FAKE_HEADROOM_OK"] = "false"

    result = _run_script(env)

    assert result.returncode == 2
    assert "disk headroom is blocked" in result.stderr
    assert "refusing post-readiness run" in result.stderr
    assert not run_root.exists()
    lines = command_log.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("live-collection-status\t")
    assert lines[1].startswith("live-collection-readiness\t")
    assert not any(line.startswith("labels-15m-v1\t") for line in lines)


def test_ready_pipeline_refuses_current_filesystem_disk_block_before_retrain(
    tmp_path: Path,
) -> None:
    env, run_root = _runner_env(tmp_path, ready=True)
    command_log = tmp_path / "commands.log"
    env["FAKE_COMMAND_LOG"] = str(command_log)
    env["FAKE_HEADROOM_OK"] = "true"
    env["FAKE_FREE_BYTES"] = str(10**15 + 1)
    env["FAKE_REQUIRED_FREE_BYTES"] = str(10**15)
    env["FAKE_MARGIN_BYTES"] = "1"

    result = _run_script(env)

    assert result.returncode == 2
    assert "current filesystem disk headroom is blocked" in result.stderr
    assert "refusing post-readiness run" in result.stderr
    assert not run_root.exists()
    lines = command_log.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("live-collection-status\t")
    assert lines[1].startswith("live-collection-readiness\t")
    assert not any(line.startswith("labels-15m-v1\t") for line in lines)


def test_ready_pipeline_skips_when_completion_sentinel_exists(tmp_path: Path) -> None:
    env, run_root = _runner_env(tmp_path, ready=True)

    first = _run_script(env)
    assert first.returncode == 0, first.stderr + first.stdout
    manifest_path = run_root / "artifacts" / "run_manifest.json"
    before = manifest_path.read_text(encoding="utf-8")

    second = _run_script(env)

    assert second.returncode == 0
    assert "post-readiness run already completed" in second.stdout
    assert manifest_path.read_text(encoding="utf-8") == before


def test_shadow_mode_refuses_to_silently_skip_when_completion_sentinel_exists(
    tmp_path: Path,
) -> None:
    env, run_root = _runner_env(tmp_path, ready=True)

    first = _run_script(env)
    assert first.returncode == 0, first.stderr + first.stdout
    manifest_path = run_root / "artifacts" / "run_manifest.json"
    before = manifest_path.read_text(encoding="utf-8")
    env["RUN_SHADOW"] = "true"
    env["SHADOW_SINCE_MS"] = "1779600000000"
    env["SHADOW_UNTIL_MS"] = "1779686400000"

    second = _run_script(env)

    assert second.returncode == 2
    assert "refusing to silently skip requested shadow/cutover work" in second.stderr
    assert manifest_path.read_text(encoding="utf-8") == before


def test_continue_mode_reuses_existing_run_for_shadow_and_bootstrap(
    tmp_path: Path,
) -> None:
    env, run_root = _runner_env(tmp_path, ready=True)

    first = _run_script(env)
    assert first.returncode == 0, first.stderr + first.stdout
    command_log = tmp_path / "commands.log"
    env["FAKE_COMMAND_LOG"] = str(command_log)
    env["CONTINUE_POST_READINESS_RUN"] = "true"
    env["RUN_SHADOW"] = "true"
    env["SHADOW_SINCE_MS"] = "1779600000000"
    env["SHADOW_UNTIL_MS"] = "1779686400000"

    second = _run_script(env)

    assert second.returncode == 0, second.stderr + second.stdout
    assert "continuing existing post-readiness run root" in second.stdout
    payload = json.loads((run_root / "artifacts" / "run_manifest.json").read_text(encoding="utf-8"))
    assert payload["phase"] == "completed"
    assert payload["settings"]["continue_post_readiness_run"] is True
    assert payload["paths"]["baseline_eval_dir"] == str(
        run_root / "artifacts" / "models" / "incumbent-same-dataset"
    )
    assert payload["paths"]["candidate_eval_dir"] == str(
        run_root / "artifacts" / "models" / "xgboost-v4-same-dataset"
    )
    lines = command_log.read_text(encoding="utf-8").splitlines()
    assert any(line.startswith("shadow-v1\t") for line in lines)
    bootstrap = next(line for line in lines if line.startswith("bootstrap-champion-v1\t"))
    assert f"--baseline-dir\t{run_root / 'artifacts' / 'models' / 'incumbent-same-dataset'}" in bootstrap
    assert f"--candidate-dir\t{run_root / 'artifacts' / 'models' / 'xgboost-v4-same-dataset'}" in bootstrap
    assert not any(line.startswith("xgboost-v4\t") for line in lines)
    assert not any(line.startswith("training-dataset-v1\t") for line in lines)


def test_ready_pipeline_refreshes_live_status_immediately_before_final_audits(
    tmp_path: Path,
) -> None:
    env, run_root = _runner_env(tmp_path, ready=True)
    command_log = tmp_path / "commands.log"
    env["FAKE_COMMAND_LOG"] = str(command_log)

    result = _run_script(env)

    assert result.returncode == 0, result.stderr + result.stdout
    lines = command_log.read_text(encoding="utf-8").splitlines()
    status_indices = [
        index for index, line in enumerate(lines) if line.startswith("live-collection-status\t")
    ]
    assert len(status_indices) >= 2
    serving_index = next(
        index for index, line in enumerate(lines) if line.startswith("serving-readiness-v1\t")
    )
    promotion_index = next(
        index for index, line in enumerate(lines) if line.startswith("champion-promotion-audit\t")
    )
    objective_indices = [
        index
        for index, line in enumerate(lines)
        if line.startswith("xgboost-v4-objective-audit\t")
    ]
    issue_indices = [
        index
        for index, line in enumerate(lines)
        if line.startswith("xgboost-v4-issue-coverage-audit\t")
    ]
    assert len(objective_indices) == 2
    assert len(issue_indices) == 2
    assert serving_index < status_indices[-1] < promotion_index < objective_indices[0]
    assert objective_indices[0] < issue_indices[0] < objective_indices[1] < issue_indices[1]
    promotion_audit = lines[promotion_index]
    objective_audit = lines[objective_indices[0]]
    issue_audit = lines[issue_indices[0]]
    assert (
        f"--candidate-eval-dir\t{run_root / 'artifacts' / 'models' / 'xgboost-v4-same-dataset'}"
        in promotion_audit
    )
    assert (
        "--promotion-process-path\t/Users/tcscoder/Downloads/champion-promotion.md"
        in promotion_audit
    )
    assert "--repo-promotion-runbook-path\tdocs/runbooks/champion_promotion.md" in promotion_audit
    assert (
        f"--candidate-model-dir\t{run_root / 'artifacts' / 'models' / 'xgboost-v4'}"
        in objective_audit
    )
    assert (
        f"--promotion-audit-path\t{run_root / 'artifacts' / 'champion-promotion-audit' / 'champion_promotion_audit.json'}"
        in objective_audit
    )
    assert (
        f"--stability-report-path\t{run_root / 'artifacts' / 'dataset-stability' / 'dataset_stability_report.json'}"
        in objective_audit
    )
    assert f"--slack-delivery-status-path\t{env['SLACK_DELIVERY_STATUS_PATH']}" in objective_audit
    assert f"--output-path\t{run_root / 'artifacts' / 'issue_coverage_audit.json'}" in issue_audit
    assert (
        f"--objective-audit-path\t{run_root / 'artifacts' / 'xgboost_v4_objective_audit.json'}"
        in issue_audit
    )
    for index in objective_indices:
        assert f"--post-readiness-latest-path\t{env['POST_READINESS_LATEST_PATH']}" in lines[index]


def test_ready_pipeline_skips_when_lock_exists(tmp_path: Path) -> None:
    env, run_root = _runner_env(tmp_path, ready=True)
    Path(env["POST_READINESS_LOCK_DIR"]).mkdir()

    result = _run_script(env)

    assert result.returncode == 0
    assert "post-readiness run already in progress" in result.stdout
    assert not run_root.exists()
    assert Path(env["POST_READINESS_LOCK_DIR"]).exists()


def test_ready_pipeline_clears_stale_lock_and_runs(tmp_path: Path) -> None:
    env, run_root = _runner_env(tmp_path, ready=True)
    lock_dir = Path(env["POST_READINESS_LOCK_DIR"])
    lock_dir.mkdir()
    metadata_path = Path(f"{env['POST_READINESS_LOCK_DIR']}.json")
    metadata_path.write_text("{}", encoding="utf-8")
    old_mtime = time.time() - 90_000
    os.utime(lock_dir, (old_mtime, old_mtime))
    os.utime(metadata_path, (old_mtime, old_mtime))

    result = _run_script(env)

    assert result.returncode == 0, result.stderr + result.stdout
    assert "removed stale post-readiness lock" in result.stdout
    assert (run_root / "artifacts" / "run_manifest.json").exists()
    assert not lock_dir.exists()
    assert not metadata_path.exists()


def test_ready_pipeline_force_rerun_ignores_completion_sentinel(tmp_path: Path) -> None:
    env, run_root = _runner_env(tmp_path, ready=True)

    first = _run_script(env)
    assert first.returncode == 0, first.stderr + first.stdout
    env["FORCE_POST_READINESS_RERUN"] = "true"

    second = _run_script(env)

    assert second.returncode == 0, second.stderr + second.stdout
    payload = json.loads((run_root / "artifacts" / "run_manifest.json").read_text(encoding="utf-8"))
    assert payload["settings"]["force_post_readiness_rerun"] is True


def test_cutover_mode_passes_existing_shadow_and_bootstrap_to_audit(tmp_path: Path) -> None:
    env, run_root = _runner_env(tmp_path, ready=True)
    env["RUN_CUTOVER_REPORT"] = "true"
    command_log = tmp_path / "commands.log"
    env["FAKE_COMMAND_LOG"] = str(command_log)

    shadow_path = run_root / "artifacts" / "shadow" / "xgboost-v4-shadow-evaluation.json"
    bootstrap_path = run_root / "artifacts" / "bootstrap" / "bootstrap_decision.json"
    smoke_path = run_root / "artifacts" / "cutover" / "inference-smoke.json"
    issue_closures_path = run_root / "artifacts" / "cutover" / "github-issue-closures.json"
    for path in (shadow_path, bootstrap_path, smoke_path, issue_closures_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")

    result = _run_script(env)

    assert result.returncode == 0, result.stderr + result.stdout
    assert "using existing shadow/bootstrap evidence for audits" in result.stdout
    lines = command_log.read_text(encoding="utf-8").splitlines()
    promotion_audit = next(line for line in lines if line.startswith("champion-promotion-audit\t"))
    assert f"--shadow-evaluation-path\t{shadow_path}" in promotion_audit
    assert f"--bootstrap-decision-path\t{bootstrap_path}" in promotion_audit
    assert (
        f"--cutover-report-path\t{run_root / 'artifacts' / 'cutover' / 'xgboost-v4-cutover.json'}"
        in promotion_audit
    )
    cutover_report = next(line for line in lines if line.startswith("champion-cutover-report-v1\t"))
    assert f"--github-issue-closures-path\t{issue_closures_path}" in cutover_report


def test_cutover_mode_uses_fresh_shadow_and_bootstrap_when_run_shadow_enabled(
    tmp_path: Path,
) -> None:
    env, run_root = _runner_env(tmp_path, ready=True)
    env["RUN_SHADOW"] = "true"
    env["SHADOW_SINCE_MS"] = "auto"
    env["SHADOW_UNTIL_MS"] = "auto"
    env["RUN_CUTOVER_REPORT"] = "true"
    command_log = tmp_path / "commands.log"
    env["FAKE_COMMAND_LOG"] = str(command_log)

    smoke_path = run_root / "artifacts" / "cutover" / "inference-smoke.json"
    issue_closures_path = run_root / "artifacts" / "cutover" / "github-issue-closures.json"
    for path in (smoke_path, issue_closures_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")

    result = _run_script(env)

    assert result.returncode == 0, result.stderr + result.stdout
    lines = command_log.read_text(encoding="utf-8").splitlines()
    shadow_index = next(index for index, line in enumerate(lines) if line.startswith("shadow-v1\t"))
    bootstrap_index = next(
        index for index, line in enumerate(lines) if line.startswith("bootstrap-champion-v1\t")
    )
    drift_index = next(
        index for index, line in enumerate(lines) if line.startswith("drift-baseline-v1\t")
    )
    cutover_index = next(
        index for index, line in enumerate(lines) if line.startswith("champion-cutover-report-v1\t")
    )
    promotion_index = next(
        index for index, line in enumerate(lines) if line.startswith("champion-promotion-audit\t")
    )
    assert shadow_index < bootstrap_index < drift_index < cutover_index < promotion_index
    shadow = lines[shadow_index]
    assert "--since-ms\t1779600000000" in shadow
    assert "--until-ms\t1779686400000" in shadow
    cutover_report = lines[cutover_index]
    shadow_path = run_root / "artifacts" / "shadow" / "xgboost-v4-shadow-evaluation.json"
    bootstrap_path = run_root / "artifacts" / "bootstrap" / "bootstrap_decision.json"
    shadow_payload = json.loads(shadow_path.read_text(encoding="utf-8"))
    assert shadow_payload["sample_count"] == 10
    assert shadow_payload["scored_count"] == 10
    assert shadow_payload["challenger_probability_distribution"]["count"] == 10
    assert shadow_payload["window_start_ts"] == 1779600000000
    assert shadow_payload["window_end_ts"] == 1779686400000
    assert shadow_payload["session_duration_seconds"] == 86400
    assert f"--shadow-evaluation-path\t{shadow_path}" in cutover_report
    assert f"--bootstrap-decision-path\t{bootstrap_path}" in cutover_report
    promotion_audit = lines[promotion_index]
    assert f"--shadow-evaluation-path\t{shadow_path}" in promotion_audit
    assert f"--bootstrap-decision-path\t{bootstrap_path}" in promotion_audit


def test_strict_final_audit_still_writes_manifest_before_failing(tmp_path: Path) -> None:
    env, run_root = _runner_env(tmp_path, ready=True)
    command_log = tmp_path / "commands.log"
    env["FAKE_COMMAND_LOG"] = str(command_log)
    env["STRICT_FINAL_AUDIT"] = "true"

    result = _run_script(env)

    assert result.returncode == 1
    assert "final audits are blocked" in result.stderr
    manifest_path = run_root / "artifacts" / "run_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["phase"] == "audit_blocked"
    assert payload["audit_results"]["promotion"]["decision"] == "BLOCKED"
    assert payload["audit_results"]["promotion"]["passed"] is False
    assert payload["audit_results"]["objective"]["decision"] == "BLOCKED"
    assert payload["audit_results"]["objective"]["objective_complete"] is False
    assert payload["audit_results"]["issue_coverage"]["exists"] is True
    lines = command_log.read_text(encoding="utf-8").splitlines()
    objective_audits = [
        line for line in lines if line.startswith("xgboost-v4-objective-audit\t")
    ]
    issue_audits = [
        line for line in lines if line.startswith("xgboost-v4-issue-coverage-audit\t")
    ]
    assert len(objective_audits) == 1
    assert len(issue_audits) == 1
    assert "--no-fail-on-blocked" in objective_audits[0]
    assert "--no-fail-on-blocked" in issue_audits[0]
    assert not list(tmp_path.rglob("*.tmp"))


def test_strict_final_audit_rejects_non_pointer_objective_blockers_before_sentinel(
    tmp_path: Path,
) -> None:
    env, run_root = _runner_env(tmp_path, ready=True)
    env["STRICT_FINAL_AUDIT"] = "true"
    env["FAKE_PROMOTION_PASSED"] = "true"

    result = _run_script(env)

    assert result.returncode == 1
    assert "non-pointer objective blockers" in result.stderr
    assert "create_xgboost_v4_model: model not final" in result.stderr
    manifest_path = run_root / "artifacts" / "run_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["phase"] == "audit_blocked"
    assert not Path(env["POST_READINESS_SENTINEL_PATH"]).exists()
    assert not Path(env["POST_READINESS_LATEST_PATH"]).exists()
    assert not Path(env["POST_READINESS_LOCK_DIR"]).exists()
    assert not list(tmp_path.rglob("*.tmp"))


def test_strict_final_audit_rejects_incomplete_latest_pointer_after_sentinel(
    tmp_path: Path,
) -> None:
    env, _ = _runner_env(tmp_path, ready=True)
    env["STRICT_FINAL_AUDIT"] = "true"
    env["FAKE_PROMOTION_PASSED"] = "true"
    env["FAKE_OBJECTIVE_ONLY_POINTER_BLOCKED"] = "true"
    env["FAKE_OBJECTIVE_COMPLETES_AFTER_POINTER"] = "true"

    result = _run_script(env)

    assert result.returncode == 1
    assert "strict final latest pointer is incomplete" in result.stderr
    assert "issue_coverage_issue_checks.#55.passed is not true" in result.stderr
    latest_path = Path(env["POST_READINESS_LATEST_PATH"])
    assert latest_path.exists()
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    assert latest["objective_complete"] is True
    assert latest["issue_coverage_issue_checks"]["#55"]["passed"] is False
    assert not list(tmp_path.rglob("*.tmp"))


def test_strict_final_audit_accepts_complete_latest_pointer(
    tmp_path: Path,
) -> None:
    env, run_root = _runner_env(tmp_path, ready=True)
    env["STRICT_FINAL_AUDIT"] = "true"
    env["FAKE_PROMOTION_PASSED"] = "true"
    env["FAKE_OBJECTIVE_ONLY_POINTER_BLOCKED"] = "true"
    env["FAKE_OBJECTIVE_COMPLETES_AFTER_POINTER"] = "true"
    env["FAKE_ISSUE_COVERAGE_PASSED"] = "true"

    result = _run_script(env)

    assert result.returncode == 0, result.stderr + result.stdout
    assert "complete; run root" in result.stdout
    latest_path = Path(env["POST_READINESS_LATEST_PATH"])
    sentinel_path = Path(env["POST_READINESS_SENTINEL_PATH"])
    assert latest_path.exists()
    assert sentinel_path.exists()
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    sentinel = json.loads(sentinel_path.read_text(encoding="utf-8"))
    for payload in (latest, sentinel):
        assert payload["run_manifest_phase"] == "completed"
        assert payload["promotion_passed"] is True
        assert payload["objective_decision"] == "COMPLETE"
        assert payload["objective_complete"] is True
        assert all(
            payload["issue_coverage_issue_checks"][issue_id]["passed"] is True
            for issue_id in ("#54", "#55", "#56", "#57", "#58", "#64", "#65")
        )
        assert all(
            criterion["passed"] is True
            for criterion in payload[
                "issue_coverage_objective_success_criteria"
            ].values()
        )
    manifest = json.loads((run_root / "artifacts" / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["phase"] == "completed"
    assert manifest["audit_results"]["objective"]["objective_complete"] is True
    assert manifest["audit_results"]["issue_coverage"]["objective_complete"] is True
    assert not list(tmp_path.rglob("*.tmp"))
