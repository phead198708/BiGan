"""Bounded observations, not a second ledger. Safe, atomic report artifacts."""

from __future__ import annotations

import json
import math
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

from .preflight import SAFETY, Preflight, validate_report_directory


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
            "dashboard_url": check.url,
            "polls": {"attempted": 0, "successful": 0, "failed": 0, "longest_failure_streak_ms": 0},
            "states": {}, "feeds": {}, "rollovers": 0, "runs_observed": [],
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
        require_finite(self.data)
        return self.data

    def markdown(self) -> str:
        d = self.data
        lines = ["# Paper stack soak — " + d["result"], "",
                 "PAPER / SIMULATED — NO REAL FUNDS", "",
                 f"Mode: `{d['mode']}`. Duration: {d['duration_ms'] / 1000:.1f} seconds.",
                 f"Source: `{d['source_commit']}`. Config: `{d['config_sha256']}`.",
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
    """Reserve an empty, disjoint artifact directory; atomically publish each file.

    An exclusive marker prevents two supervisors overwriting the same report.
    A crash can leave that marker for explicit operator inspection, never repair.
    No handles into the paper output tree are opened here.
    """

    def __init__(self, directory: Path, *, output: Path) -> None:
        validate_report_directory(directory, output)
        directory.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            claim = os.open(".soak-report.lock", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=self.fd)
            os.close(claim)
            if set(os.listdir(self.fd)) != {".soak-report.lock"}:
                raise ValueError("REPORT_DIRECTORY_NOT_EMPTY")
        except BaseException:
            os.close(self.fd)
            self.fd = -1
            raise

    def write(self, report: SoakReport) -> None:
        self._atomic("soak_report.json", json.dumps(report.data, allow_nan=False, indent=2) + "\n")
        self._atomic("soak_summary.md", report.markdown())
        os.fsync(self.fd)

    def _atomic(self, name: str, text: str) -> None:
        temporary = "." + uuid.uuid4().hex + ".tmp"
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=self.fd)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            # Never replace somebody else's previous artifact.
            if name in os.listdir(self.fd):
                raise ValueError("REPORT_ALREADY_EXISTS")
            os.replace(temporary, name, src_dir_fd=self.fd, dst_dir_fd=self.fd)
        finally:
            if temporary in os.listdir(self.fd):
                os.unlink(temporary, dir_fd=self.fd)

    def close(self) -> None:
        if self.fd >= 0:
            os.unlink(".soak-report.lock", dir_fd=self.fd)
            os.close(self.fd)
            self.fd = -1
