import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check_xgboost_v4_collection_risk.sh"


def _status_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_status(
    path: Path,
    *,
    headroom_ok: bool,
    free_gib: int = 50,
    generated_at: str | None = None,
) -> None:
    free_bytes = free_gib * 1024**3
    required_bytes = 45 * 1024**3
    margin_bytes = free_bytes - required_bytes
    payload = {
        "generated_at": generated_at or _status_timestamp(),
        "screen_session": "xgbv4_7d_atomic_20260523T125657Z",
        "screen_state": "running",
        "collection_readiness": {
            "ready_for_training": False,
            "estimated_ready_at": "2026-05-31T03:00:00Z",
            "features_15m_v1": {
                "target_progress_pct": 18.2,
                "remaining_target_days": 5.72,
                "limiting_family": "BTC-15M",
            },
            "labels_15m_v1": {
                "target_progress_pct": 18.1,
                "remaining_target_days": 5.73,
                "limiting_family": "BTC-15M",
            },
            "quarantine_clean_window": {
                "quarantined_count": 1,
                "target_progress_pct": 9.79,
                "remaining_target_days": 6.31,
                "estimated_ready_at": "2026-05-31T03:00:00Z",
                "latest_quarantined_segment": {
                    "path": "raw_invalid/ws_market/2026-05-24T030000Z.ndjson.gz"
                },
            },
        },
        "liveness_evidence": {
            "raw_segments_fresh": True,
            "processed_manifest_fresh": True,
            "latest_raw_segment": {"age_seconds": 70},
            "latest_processed_segment": {"age_seconds": 250},
        },
        "raw_segment_integrity": {"invalid_count": 0},
        "health_evidence": {"unrecovered_error_match_count": 0},
        "disk_headroom_evidence": {
            "headroom_ok": headroom_ok,
            "headroom_low_margin": True,
            "free_bytes": free_bytes,
            "required_free_bytes": required_bytes,
            "projected_remaining_bytes": required_bytes,
            "headroom_margin_bytes": margin_bytes,
            "low_margin_threshold_bytes": 10 * 1024**3,
            "min_free_bytes": 5 * 1024**3,
            "live_root_size_bytes": 7 * 1024**3,
            "observed_raw_span_days": 1.75,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _run_script(
    tmp_path: Path,
    status_path: Path,
    live_root: Path,
    *args: str,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "HOME": str(tmp_path / "home"),
        "STATUS_PATH": str(status_path),
        "LIVE_ROOT": str(live_root),
        "SHOW_LIVE_ROOTS": "false",
        **(extra_env or {}),
    }
    (tmp_path / "home").mkdir(parents=True)
    (tmp_path / "home" / "Library" / "Caches").mkdir(parents=True)
    return subprocess.run(
        [str(SCRIPT), *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_collection_risk_reports_false_readiness_and_low_margin(tmp_path: Path) -> None:
    status_path = tmp_path / "status.json"
    live_root = tmp_path / "live-root"
    live_root.mkdir()
    _write_status(status_path, headroom_ok=True)

    result = _run_script(tmp_path, status_path, live_root)

    assert result.returncode == 0
    assert "ready_for_training=false" in result.stdout
    assert "fresh=true" in result.stdout
    assert "headroom_ok=true headroom_low_margin=true" in result.stdout
    assert "reclaim_to_clear_block=0.00GiB reclaim_to_clear_low_margin=5.00GiB" in result.stdout
    assert "urgency_estimate growth_per_day=4.00GiB min_free=5.00GiB estimated_days_to_ready=6.310 status_days_to_min_free=11.250" in result.stdout
    assert "min_free_before_ready status=false" in result.stdout
    assert "current_fs_free=" in result.stdout
    assert "current_reclaim_to_clear_block=" in result.stdout
    assert "Suggested reclaim target to clear the low-margin buffer from status artifact: 5.00GiB" in result.stdout
    assert "Suggested reclaim target to clear the low-margin buffer from current filesystem:" in result.stdout
    assert "WARNING: disk headroom passes but margin is low" in result.stdout


def test_collection_risk_exits_nonzero_when_headroom_blocked(tmp_path: Path) -> None:
    status_path = tmp_path / "status.json"
    live_root = tmp_path / "live-root"
    live_root.mkdir()
    _write_status(status_path, headroom_ok=False, free_gib=20)

    result = _run_script(tmp_path, status_path, live_root)

    assert result.returncode == 2
    assert "ERROR: disk headroom is blocked" in result.stdout
    assert "reclaim_to_clear_block=25.00GiB reclaim_to_clear_low_margin=35.00GiB" in result.stdout
    assert "urgency_estimate growth_per_day=4.00GiB min_free=5.00GiB estimated_days_to_ready=6.310 status_days_to_min_free=3.750" in result.stdout
    assert "min_free_before_ready status=true" in result.stdout
    assert "current_fs_free=" in result.stdout
    assert "current_reclaim_to_clear_low_margin=" in result.stdout
    assert "ACTION NEEDED (status artifact): reclaim at least 25.00GiB to clear the hard block, or 35.00GiB to clear the low-margin buffer" in result.stdout
    assert "ACTION NEEDED (current filesystem): reclaim at least" in result.stdout
    assert "get explicit approval before pruning Docker" in result.stdout


def test_collection_risk_json_reports_low_margin_evidence(tmp_path: Path) -> None:
    status_path = tmp_path / "status.json"
    live_root = tmp_path / "live-root"
    live_root.mkdir()
    _write_status(status_path, headroom_ok=True)

    result = _run_script(tmp_path, status_path, live_root, "--json")

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["status_path"] == str(status_path)
    assert payload["status_artifact"]["fresh"] is True
    assert payload["status_artifact"]["max_age_seconds"] == 1800
    assert isinstance(payload["status_artifact"]["age_seconds"], int)
    assert payload["status_level"] == "warning"
    assert payload["blocked"] is False
    assert payload["exit_code"] == 0
    assert payload["readiness"]["ready_for_training"] is False
    assert payload["readiness"]["features_15m_v1"]["target_progress_pct"] == 18.2
    assert payload["readiness"]["features_15m_v1"]["remaining_target_days"] == 5.72
    assert payload["readiness"]["features_15m_v1"]["limiting_family"] == "BTC-15M"
    assert payload["liveness"]["raw_segments_fresh"] is True
    assert payload["health"]["invalid_recent_gzip_count"] == 0
    assert payload["disk_headroom"]["headroom_ok"] is True
    assert payload["disk_headroom"]["min_free_bytes"] == 5 * 1024**3
    assert payload["disk_headroom"]["reclaim_to_clear_block_bytes"] == 0
    assert payload["disk_headroom"]["reclaim_to_clear_low_margin_bytes"] == 5 * 1024**3
    assert payload["disk_urgency"]["observed_raw_span_days"] == 1.75
    assert payload["disk_urgency"]["estimated_growth_bytes_per_day"] == 4 * 1024**3
    assert payload["disk_urgency"]["estimated_days_to_ready"] == 6.31
    assert payload["disk_urgency"]["status_days_to_min_free"] == 11.25
    assert payload["disk_urgency"]["status_min_free_before_ready"] is False
    assert isinstance(payload["disk_urgency"]["current_filesystem_days_to_min_free"], float)
    assert isinstance(
        payload["disk_urgency"]["current_filesystem_min_free_before_ready"], bool
    )
    current = payload["current_filesystem_headroom"]
    assert current["live_root"] == str(live_root)
    assert isinstance(current["free_bytes"], int)
    assert isinstance(current["reclaim_to_clear_block_bytes"], int)
    assert isinstance(current["reclaim_to_clear_low_margin_bytes"], int)
    assert isinstance(payload["reclaim_candidates"], list)
    assert any(candidate["label"] == "user caches" for candidate in payload["reclaim_candidates"])


def test_collection_risk_json_reports_blocked_status_artifact(tmp_path: Path) -> None:
    status_path = tmp_path / "status.json"
    live_root = tmp_path / "live-root"
    live_root.mkdir()
    _write_status(status_path, headroom_ok=False, free_gib=20)

    result = _run_script(tmp_path, status_path, live_root, "--json")

    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status_level"] == "blocked"
    assert payload["blocked"] is True
    assert payload["exit_code"] == 2
    assert payload["disk_headroom"]["headroom_ok"] is False
    assert payload["disk_headroom"]["headroom_margin_bytes"] == -(25 * 1024**3)
    assert payload["disk_headroom"]["reclaim_to_clear_block_bytes"] == 25 * 1024**3
    assert payload["disk_headroom"]["reclaim_to_clear_low_margin_bytes"] == 35 * 1024**3
    assert payload["disk_urgency"]["status_days_to_min_free"] == 3.75
    assert payload["disk_urgency"]["status_min_free_before_ready"] is True


def test_collection_risk_json_writes_output_path_atomically(tmp_path: Path) -> None:
    status_path = tmp_path / "status.json"
    live_root = tmp_path / "live-root"
    output_path = tmp_path / "artifacts" / "risk.json"
    live_root.mkdir()
    _write_status(status_path, headroom_ok=False, free_gib=20)

    result = _run_script(
        tmp_path,
        status_path,
        live_root,
        "--output-path",
        str(output_path),
    )

    assert result.returncode == 2
    stdout_payload = json.loads(result.stdout)
    file_payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert file_payload == stdout_payload
    assert file_payload["status_level"] == "blocked"
    assert file_payload["disk_headroom"]["reclaim_to_clear_block_bytes"] == 25 * 1024**3


def test_collection_risk_warns_when_status_artifact_is_stale(tmp_path: Path) -> None:
    status_path = tmp_path / "status.json"
    live_root = tmp_path / "live-root"
    live_root.mkdir()
    _write_status(
        status_path,
        headroom_ok=True,
        generated_at="2026-05-24T00:00:00Z",
    )

    result = _run_script(
        tmp_path,
        status_path,
        live_root,
        "--json",
        extra_env={"STATUS_MAX_AGE_SECONDS": "1"},
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["status_level"] == "warning"
    assert payload["status_artifact"]["fresh"] is False
    assert payload["status_artifact"]["max_age_seconds"] == 1

    human = _run_script(
        tmp_path / "human",
        status_path,
        live_root,
        extra_env={"STATUS_MAX_AGE_SECONDS": "1"},
    )
    assert human.returncode == 0
    assert "WARNING: status artifact is stale" in human.stdout
