"""Exclusive, audited handoff between sequential collection supervisors."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_pairwise_action_advantage_lcb import (
    FORBIDDEN_REGISTRY_FIELDS,
    _find_fields,
)

SCHEMA_PREFIX = "bigan-v8-hybrid-pairwise-collection-supervisor-handoff"
CLAIM_SCHEMA_VERSION = f"{SCHEMA_PREFIX}-claim-v1"
REPORT_SCHEMA_VERSION = f"{SCHEMA_PREFIX}-report-v1"
MANIFEST_SCHEMA_VERSION = f"{SCHEMA_PREFIX}-manifest-v1"


@dataclass(frozen=True, slots=True)
class ProcessIdentity:
    """Expected PID identity and script hash for a handoff participant."""

    role: str
    pid: int
    required_command_substring: str
    script_path: Path | str | None = None
    expected_script_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.role.strip():
            raise ValueError("process role is required")
        if self.pid <= 1:
            raise ValueError(f"{self.role} PID must be greater than 1")
        if not self.required_command_substring.strip():
            raise ValueError(
                f"{self.role} required command substring is required"
            )
        if (self.script_path is None) != (
            self.expected_script_sha256 is None
        ):
            raise ValueError(
                f"{self.role} script path and SHA-256 must be provided together"
            )
        if self.script_path is not None:
            object.__setattr__(self, "script_path", Path(self.script_path))
            _require_sha256(
                str(self.expected_script_sha256),
                name=f"{self.role} script SHA-256",
            )


@dataclass(frozen=True, slots=True)
class HybridCollectionSupervisorHandoffConfig:
    """Inputs for one exclusive operational supervisor handoff."""

    run_id: str
    output_dir: Path | str
    claim_path: Path | str
    batch_progress_path: Path | str
    expected_batch_id: str
    observed_at_ts: int
    superseded_supervisor: ProcessIdentity
    successor_supervisor: ProcessIdentity
    protected_processes: tuple[ProcessIdentity, ...]
    apply_termination: bool = False
    termination_wait_seconds: float = 10.0
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if not self.expected_batch_id.strip():
            raise ValueError("expected_batch_id is required")
        if self.observed_at_ts <= 0:
            raise ValueError("observed_at_ts must be positive")
        if self.termination_wait_seconds <= 0.0:
            raise ValueError("termination_wait_seconds must be positive")
        if not self.protected_processes:
            raise ValueError("protected_processes must not be empty")
        protected_pids = {process.pid for process in self.protected_processes}
        if len(protected_pids) != len(self.protected_processes):
            raise ValueError("protected process PIDs must be unique")
        if self.superseded_supervisor.pid in protected_pids:
            raise ValueError(
                "superseded supervisor PID cannot be a protected PID"
            )
        if self.successor_supervisor.pid in protected_pids:
            raise ValueError(
                "successor supervisor must be listed only once"
            )
        if self.superseded_supervisor.pid == self.successor_supervisor.pid:
            raise ValueError(
                "superseded and successor supervisor PIDs must differ"
            )
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "claim_path", Path(self.claim_path))
        object.__setattr__(
            self,
            "batch_progress_path",
            Path(self.batch_progress_path),
        )


def perform_exclusive_collection_supervisor_handoff(
    config: HybridCollectionSupervisorHandoffConfig,
) -> dict[str, Any]:
    """Claim one successor and optionally terminate only the superseded waiter."""

    run_dir = (config.output_dir / config.run_id).expanduser().resolve()
    if run_dir.exists():
        if not config.overwrite_existing:
            raise ValueError(f"run directory already exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    batch_path = config.batch_progress_path.expanduser().resolve()
    if not batch_path.is_file():
        raise ValueError(f"batch progress is missing: {batch_path}")
    batch_before = _load_json(batch_path)
    _validate_batch_progress(
        batch_before,
        expected_batch_id=config.expected_batch_id,
    )
    batch_before_descriptor = _descriptor(batch_path)
    process_expectations = (
        config.superseded_supervisor,
        config.successor_supervisor,
        *config.protected_processes,
    )
    before_snapshots = [
        _validated_process_snapshot(expectation)
        for expectation in process_expectations
    ]
    expectation_by_role = {
        expectation.role: expectation for expectation in process_expectations
    }
    if len(expectation_by_role) != len(process_expectations):
        raise ValueError("process roles must be unique")

    claim = {
        "schema_version": CLAIM_SCHEMA_VERSION,
        "run_id": config.run_id,
        "claimed_at_ts": config.observed_at_ts,
        "expected_batch_id": config.expected_batch_id,
        "batch_progress_before": batch_before_descriptor,
        "superseded_supervisor": _process_expectation_descriptor(
            config.superseded_supervisor
        ),
        "selected_successor_supervisor": _process_expectation_descriptor(
            config.successor_supervisor
        ),
        "protected_processes": [
            _process_expectation_descriptor(process)
            for process in config.protected_processes
        ],
        "termination_scope": "superseded_supervisor_only",
        "apply_termination": config.apply_termination,
        "labels_outcomes_pnl_or_validation_metrics_opened": False,
        "collector_or_batch_artifacts_mutated": False,
        **_safety_fields(),
    }
    claim["claim_id"] = canonical_json_sha256(claim)
    claim_path = config.claim_path.expanduser().resolve()
    _write_exclusive_json(claim_path, claim)

    termination_attempted = False
    termination_signal = None
    if config.apply_termination:
        termination_attempted = True
        termination_signal = "SIGTERM"
        _terminate_pid(
            config.superseded_supervisor.pid,
            wait_seconds=config.termination_wait_seconds,
        )

    after_snapshots = []
    superseded_alive_after = _pid_alive(
        config.superseded_supervisor.pid
    )
    if not config.apply_termination:
        after_snapshots.append(
            _validated_process_snapshot(config.superseded_supervisor)
        )
    elif superseded_alive_after:
        raise RuntimeError(
            "superseded supervisor remained alive after bounded SIGTERM wait"
        )
    for expectation in (
        config.successor_supervisor,
        *config.protected_processes,
    ):
        after_snapshots.append(_validated_process_snapshot(expectation))

    batch_after = _load_json(batch_path)
    _validate_batch_progress(
        batch_after,
        expected_batch_id=config.expected_batch_id,
    )
    if int(batch_after.get("capture_count") or 0) < int(
        batch_before.get("capture_count") or 0
    ):
        raise RuntimeError("batch capture count regressed during handoff")
    if int(batch_after.get("error_count") or 0) < int(
        batch_before.get("error_count") or 0
    ):
        raise RuntimeError("batch error count regressed during handoff")
    batch_after_descriptor = _descriptor(batch_path)
    handoff_applied = (
        config.apply_termination
        and not superseded_alive_after
        and all(snapshot["alive"] for snapshot in after_snapshots)
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run_id": config.run_id,
        "status": (
            "exclusive_supervisor_handoff_applied"
            if handoff_applied
            else "dry_run_exclusive_handoff_validated"
        ),
        "claim": _descriptor(claim_path),
        "claim_id": claim["claim_id"],
        "expected_batch_id": config.expected_batch_id,
        "batch_progress_before": batch_before_descriptor,
        "batch_progress_after": batch_after_descriptor,
        "batch_progress_hash_changed_during_handoff": (
            batch_before_descriptor["sha256"]
            != batch_after_descriptor["sha256"]
        ),
        "capture_count_before": int(
            batch_before.get("capture_count") or 0
        ),
        "capture_count_after": int(
            batch_after.get("capture_count") or 0
        ),
        "error_count_before": int(batch_before.get("error_count") or 0),
        "error_count_after": int(batch_after.get("error_count") or 0),
        "process_snapshots_before": before_snapshots,
        "process_snapshots_after": after_snapshots,
        "superseded_supervisor_pid": config.superseded_supervisor.pid,
        "successor_supervisor_pid": config.successor_supervisor.pid,
        "protected_process_pids": [
            process.pid for process in config.protected_processes
        ],
        "termination_scope": "superseded_supervisor_only",
        "termination_attempted": termination_attempted,
        "termination_signal": termination_signal,
        "superseded_supervisor_alive_after": superseded_alive_after,
        "successor_supervisor_alive_after": _pid_alive(
            config.successor_supervisor.pid
        ),
        "all_protected_processes_alive_after": all(
            _pid_alive(process.pid) for process in config.protected_processes
        ),
        "exactly_one_post_batch_successor_selected": True,
        "handoff_applied": handoff_applied,
        "collector_or_batch_artifacts_mutated": False,
        "labels_outcomes_pnl_or_validation_metrics_opened": False,
        "model_training_or_prediction_attempted": False,
        "source_scores_mutated": False,
        "execution_thresholds_mutated": False,
        **_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    report_path = (
        run_dir / "hybrid_pairwise_collection_supervisor_handoff_report.json"
    )
    markdown_path = (
        run_dir / "hybrid_pairwise_collection_supervisor_handoff_report.md"
    )
    _write_json(report_path, report)
    markdown_path.write_text(_report_markdown(report), encoding="utf-8")
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "claim": _descriptor(claim_path),
        "handoff_report": _descriptor(report_path),
        "handoff_report_markdown": _descriptor(markdown_path),
        "batch_progress_before": batch_before_descriptor,
        "batch_progress_after": batch_after_descriptor,
        "handoff_applied": handoff_applied,
        "superseded_supervisor_alive_after": superseded_alive_after,
        "successor_supervisor_alive_after": report[
            "successor_supervisor_alive_after"
        ],
        "all_protected_processes_alive_after": report[
            "all_protected_processes_alive_after"
        ],
        "exactly_one_post_batch_successor_selected": True,
        "collector_or_batch_artifacts_mutated": False,
        "labels_outcomes_pnl_or_validation_metrics_opened": False,
        **_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = (
        run_dir / "hybrid_pairwise_collection_supervisor_handoff_manifest.json"
    )
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "claim_path": claim_path,
        "claim_sha256": _sha256_file(claim_path),
        "report_path": report_path,
        "report_sha256": _sha256_file(report_path),
        "markdown_path": markdown_path,
        "markdown_sha256": _sha256_file(markdown_path),
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
        "claim": claim,
        "report": report,
        "manifest": manifest,
    }


def _validated_process_snapshot(
    expectation: ProcessIdentity,
) -> dict[str, Any]:
    snapshot = _process_snapshot(expectation.pid)
    if not snapshot["alive"]:
        raise RuntimeError(f"{expectation.role} PID is not alive")
    if expectation.required_command_substring not in snapshot["command"]:
        raise RuntimeError(
            f"{expectation.role} PID command identity mismatch"
        )
    script_descriptor = None
    if expectation.script_path is not None:
        script_path = expectation.script_path.expanduser().resolve()
        if not script_path.is_file():
            raise ValueError(
                f"{expectation.role} script is missing: {script_path}"
            )
        assert expectation.expected_script_sha256 is not None
        if _sha256_file(script_path) != expectation.expected_script_sha256:
            raise ValueError(
                f"{expectation.role} script SHA-256 mismatch"
            )
        script_descriptor = _descriptor(script_path)
    return {
        **snapshot,
        "role": expectation.role,
        "required_command_substring": expectation.required_command_substring,
        "script": script_descriptor,
        "identity_verified": True,
    }


def _process_snapshot(pid: int) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ps",
            "-p",
            str(pid),
            "-o",
            "pid=",
            "-o",
            "ppid=",
            "-o",
            "etime=",
            "-o",
            "stat=",
            "-o",
            "command=",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    output = completed.stdout.strip()
    if completed.returncode != 0 or not output:
        return {
            "pid": pid,
            "ppid": None,
            "elapsed": None,
            "stat": None,
            "command": "",
            "alive": False,
        }
    fields = output.split(maxsplit=4)
    if len(fields) < 5:
        raise RuntimeError(f"unable to parse process snapshot for PID {pid}")
    return {
        "pid": int(fields[0]),
        "ppid": int(fields[1]),
        "elapsed": fields[2],
        "stat": fields[3],
        "command": fields[4],
        "alive": True,
    }


def _terminate_pid(pid: int, *, wait_seconds: float) -> None:
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + wait_seconds
    while _pid_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _process_expectation_descriptor(
    process: ProcessIdentity,
) -> dict[str, Any]:
    return {
        "role": process.role,
        "pid": process.pid,
        "required_command_substring": process.required_command_substring,
        "script": (
            None
            if process.script_path is None
            else {
                "path": str(process.script_path.expanduser().resolve()),
                "sha256": process.expected_script_sha256,
            }
        ),
    }


def _validate_batch_progress(
    payload: dict[str, Any],
    *,
    expected_batch_id: str,
) -> None:
    forbidden = sorted(_find_fields(payload, FORBIDDEN_REGISTRY_FIELDS))
    if forbidden:
        raise ValueError(
            "batch progress contains forbidden outcome fields: "
            + ", ".join(forbidden)
        )
    if payload.get("batch_id") != expected_batch_id:
        raise ValueError("batch progress identity mismatch")
    if (
        payload.get("paper_only") is not True
        or payload.get("capital_at_risk") is not False
    ):
        raise ValueError("batch progress safety contract failed")
    captures = list(payload.get("captures") or [])
    if int(payload.get("capture_count") or 0) != len(captures):
        raise ValueError("batch progress capture count mismatch")
    errors = list(payload.get("errors") or [])
    if int(payload.get("error_count") or 0) != len(errors):
        raise ValueError("batch progress error count mismatch")


def _write_exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise ValueError(
            f"exclusive handoff claim already exists: {path}"
        ) from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _report_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Hybrid Collection Supervisor Handoff",
            "",
            f"- status: `{report['status']}`",
            (
                "- handoff applied: "
                f"`{str(report['handoff_applied']).lower()}`"
            ),
            (
                "- superseded supervisor alive after: "
                f"`{str(report['superseded_supervisor_alive_after']).lower()}`"
            ),
            (
                "- successor supervisor alive after: "
                f"`{str(report['successor_supervisor_alive_after']).lower()}`"
            ),
            (
                "- all protected processes alive after: "
                f"`{str(report['all_protected_processes_alive_after']).lower()}`"
            ),
            "- termination scope: `superseded_supervisor_only`",
            "- collector or batch artifacts mutated: `false`",
            "- labels/outcomes/PnL/validation metrics opened: `false`",
            "- paper/live/write/wallet/capital unlock: `false`",
            "",
        ]
    )


def _safety_fields() -> dict[str, Any]:
    return {
        "paper_only": True,
        "capital_at_risk": False,
        "broker_exchange_write_enabled": False,
        "live_exchange_write_enabled": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "v8_execution_handoff_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
    }


def _require_sha256(value: str, *, name: str) -> None:
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} must be a SHA-256 hex digest")


def _descriptor(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256_file(path)}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
