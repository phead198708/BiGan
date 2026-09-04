"""Bounded observations, not a second ledger. Safe, atomic report artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any

from .preflight import SAFETY, Preflight, validate_report_directory

INCOMPLETE_FILE = ".soak-report.lock"
COMPLETE_FILE = "soak_complete.json"
ARTIFACTS = ("soak_report.json", "soak_summary.md")


def now_ms() -> int:
    return time.time_ns() // 1_000_000


def require_finite(value: Any, depth: int = 0) -> None:
    if depth > 32:
        raise ValueError("SCHEMA_DEPTH_EXCEEDED")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            valid = math.isfinite(value)
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError("NONFINITE_API_NUMBER")
    elif isinstance(value, dict):
        for item in value.values():
            require_finite(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            require_finite(item, depth + 1)


class SoakReport:
    """Only allowlisted scalar observations survive a poll; never payloads/logs."""

    def __init__(self, check: Preflight, *, started_at_ms: int | None = None) -> None:
        self.data: dict[str, Any] = {
            "schema_version": 1, "started_at_ms": now_ms() if started_at_ms is None else started_at_ms,
            "ended_at_ms": None, "duration_ms": 0, "result": "PASS", "mode": check.mode,
            "paper_safety": dict(SAFETY), "operator_identity": check.identity,
            "source_commit": check.config.source_commit, "config_sha256": check.config.config_sha256,
            "build_provenance": check.build_provenance, "publication_schema_version": 1,
            "market_data_source": check.config.binance_source_identity(),
            "dashboard_url": check.url,
            "polls": {"attempted": 0, "successful": 0, "failed": 0, "longest_failure_streak_ms": 0},
            "states": {}, "feeds": {}, "rollovers": 0, "runs_observed": [],
            "live_readiness": {"ready_samples": 0, "unready_samples": 0},
            "requested_duration_ms": None, "measurement_duration_ms": 0,
            "account": dict.fromkeys((
                "initial_equity", "final_equity", "final_cash", "realized_pnl", "unrealized_pnl",
                "fees", "max_observed_drawdown",
            )),
            "activity": dict.fromkeys(("decisions", "fills", "rejects", "holds", "settlements"), 0),
            "warnings": [], "hard_failures": [], "final_state": None,
            "scope": "Sampled HTTP observations, not ledger audit or authoritative PnL; activity is a lower bound.",
        }

    def issue(self, code: str, *, hard: bool = False, informational: bool = False) -> None:
        # No externally sourced strings (paths, titles, errors, HTTP bodies).
        if not re.fullmatch(r"[A-Z][A-Z0-9_]{0,79}", code):
            raise ValueError("invalid diagnostic code")
        key = "hard_failures" if hard else "warnings"
        items = self.data[key]
        for item in items:
            if item["code"] == code:
                item["count"] += 1
                if not informational and not hard:
                    item["severity"] = "warning"
                break
        else:
            if len(items) < 64:
                items.append({"code": code, "count": 1,
                              "severity": "failure" if hard else "info" if informational else "warning"})
        if hard:
            self.data["result"] = "FAIL"
        elif not informational and self.data["result"] != "FAIL":
            self.data["result"] = "WARN"

    @property
    def failed(self) -> bool:
        return self.data["result"] == "FAIL"

    def finish(self, *, ended_at_ms: int | None = None) -> dict[str, Any]:
        end = now_ms() if ended_at_ms is None else ended_at_ms
        self.data["ended_at_ms"] = end
        self.data["duration_ms"] = max(0, end - self.data["started_at_ms"])
        for key in ("decisions", "fills", "settlements"):
            if not self.data["activity"][key]:
                self.issue(f"NO_{key.upper()}_OBSERVED", informational=True)
        if not self.data["polls"]["successful"]:
            self.issue("NO_VALID_OBSERVATIONS", hard=True)
        if self.data["mode"] == "live_public_feeds_paper_execution":
            if not self.data["live_readiness"]["ready_samples"]:
                self.issue("LIVE_INPUTS_NEVER_READY", hard=True)
            requested = self.data["requested_duration_ms"]
            if requested is not None and self.data["measurement_duration_ms"] < requested:
                self.issue("LIVE_DURATION_NOT_COMPLETED", hard=True)
        require_finite(self.data)
        return self.data

    def markdown(self) -> str:
        d = self.data
        lines = ["# Paper stack soak — " + d["result"], "",
                 "PAPER / SIMULATED — NO REAL FUNDS", "",
                 f"Mode: `{d['mode']}`. Duration: {d['duration_ms'] / 1000:.1f} seconds.",
                 f"Measurement: {d['measurement_duration_ms'] / 1000:.1f} seconds after readiness "
                 f"(requested ms: {d['requested_duration_ms']}).",
                 f"Live input samples: {d['live_readiness']}.",
                 f"Source: `{d['source_commit']}`. Config: `{d['config_sha256']}`.",
                 f"Market data: {d['market_data_source']['display_name']} / "
                 f"`{d['market_data_source']['symbol']}` / `{d['market_data_source']['source']}`.",
                 f"Final state: `{d['final_state']}`. Observed rollovers: {d['rollovers']}.", "",
                 f"Polls: {d['polls']['successful']} successful / {d['polls']['attempted']} attempted.",
                 "", "Canonical account observations:", ""]
        lines += [f"- {key}: {value}" for key, value in d["account"].items()]
        lines += ["", "Activity observations (lower bound):", ""]
        lines += [f"- {key}: {value}" for key, value in d["activity"].items()]
        lines += ["", "Diagnostics:", ""]
        lines += [f"- {x['severity']}: `{x['code']}` ({x['count']})"
                  for x in d["hard_failures"] + d["warnings"]] or ["- None."]
        lines += ["", d["scope"], "",
                  "No wallet, signing, private exchange credentials, real order path or real funds.", ""]
        return "\n".join(lines)


class ReportWriter:
    """Stage all artifacts; atomically rename the incomplete marker to completion.

    ONLY a completion manifest with matching hashes is a published report. An
    exception invalidates this writer's public artifacts and keeps an incomplete
    marker; closing a descriptor never turns a failed publication into success.
    """

    def __init__(self, directory: Path, *, output: Path) -> None:
        validate_report_directory(directory, output)
        directory.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        self._temporary: set[str] = set()
        self._published: set[str] = set()
        self._attempted = False
        try:
            claim = os.open(INCOMPLETE_FILE, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=self.fd)
            try:
                os.fsync(claim)
            finally:
                os.close(claim)
            os.fsync(self.fd)
            if set(os.listdir(self.fd)) != {INCOMPLETE_FILE}:
                raise ValueError("REPORT_DIRECTORY_NOT_EMPTY")
        except BaseException:
            os.close(self.fd)
            self.fd = -1
            raise

    def write(self, report: SoakReport) -> None:
        if self._attempted:
            raise ValueError("REPORT_PUBLICATION_ALREADY_ATTEMPTED")
        self._attempted = True
        try:
            contents = {
                "soak_report.json": (json.dumps(report.data, allow_nan=False, indent=2) + "\n").encode(),
                "soak_summary.md": report.markdown().encode(),
            }
            manifest = {"schema_version": 1, "result": report.data["result"],
                        "files": {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}}
            # Every byte is staged and fsync'd BEFORE any artifact is published.
            staged = {name: self._stage(data) for name, data in contents.items()}
            marker = self._stage((json.dumps(manifest, sort_keys=True) + "\n").encode())
            for name, temporary in staged.items():
                if name in os.listdir(self.fd):
                    raise ValueError("REPORT_ALREADY_EXISTS")
                self._published.add(name)
                self._replace(temporary, name)
            # The owned marker is still INCOMPLETE even when it contains the
            # future manifest. One final rename removes it and commits the pair.
            self._replace(marker, INCOMPLETE_FILE)
            os.fsync(self.fd)
            self._published.add(COMPLETE_FILE)
            self._replace(INCOMPLETE_FILE, COMPLETE_FILE)
            os.fsync(self.fd)
        except BaseException:
            report.issue("REPORT_WRITE_FAILED", hard=True)
            self._abort()
            raise

    def _stage(self, contents: bytes) -> str:
        temporary = "." + uuid.uuid4().hex + ".tmp"
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=self.fd)
        self._temporary.add(temporary)
        with os.fdopen(fd, "wb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        return temporary

    def _replace(self, source: str, destination: str) -> None:
        os.replace(source, destination, src_dir_fd=self.fd, dst_dir_fd=self.fd)
        self._temporary.discard(source)

    def _abort(self) -> None:
        # Only files this instance created/attempted to publish are invalidated.
        # Even a persistent I/O outage cannot make a remaining candidate valid:
        # consumers MUST validate the completion marker and reject INCOMPLETE.
        with suppress(OSError):
            claim = os.open(INCOMPLETE_FILE, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=self.fd)
            try:
                os.fsync(claim)
            finally:
                os.close(claim)
        for name in self._published | self._temporary:
            with suppress(OSError):
                os.unlink(name, dir_fd=self.fd)
        with suppress(OSError):
            os.fsync(self.fd)

    def close(self) -> None:
        """Release only the read-only directory FD; never change publication.

        Retire the descriptor before closing: after a close error its state is
        uncertain, and retrying could close an unrelated, reused descriptor.
        """
        if self.fd >= 0:
            fd = self.fd
            self.fd = -1
            os.close(fd)


def load_completed_report(directory: Path) -> dict[str, Any]:
    """Machine-consumer boundary: neither JSON alone nor marker alone is enough."""
    try:
        if (directory / INCOMPLETE_FILE).exists():
            raise ValueError("incomplete")
        for name in (*ARTIFACTS, COMPLETE_FILE):
            path = directory / name
            if path.is_symlink() or path.stat().st_size > 2_000_000:
                raise ValueError("invalid artifact")
        encoded = (directory / COMPLETE_FILE).read_bytes()
        manifest = json.loads(encoded)
        if (set(manifest) != {"schema_version", "result", "files"}
                or type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1
                or manifest["result"] not in {"PASS", "WARN", "FAIL"}
                or set(manifest["files"]) != set(ARTIFACTS)):
            raise ValueError("invalid completion manifest")
        contents = {name: (directory / name).read_bytes() for name in ARTIFACTS}
        if any(hashlib.sha256(data).hexdigest() != manifest["files"][name] for name, data in contents.items()):
            raise ValueError("artifact hash mismatch")
        report = json.loads(contents["soak_report.json"])
        require_finite(report)
        if (report["publication_schema_version"] != 1 or report["result"] != manifest["result"]
                or (directory / INCOMPLETE_FILE).exists()
                or (directory / COMPLETE_FILE).read_bytes() != encoded):
            raise ValueError("publication changed")
        return report
    except (OSError, ValueError, TypeError, KeyError, RecursionError):
        raise ValueError("REPORT_NOT_COMPLETE") from None
