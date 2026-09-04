"""Pure validation. No directory creation, writer ownership or outbound I/O."""

from __future__ import annotations

import re
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.build_provenance import BuildProvenanceError, require_source_commit
from bigan.paper_trading.dashboard.server import loopback_host
from bigan.paper_trading.operator.config import OperatorConfig, load_operator_config

SAFETY = {
    "paper_only": True,
    "capital_at_risk": False,
    "broker_exchange_write_enabled": False,
    "live_exchange_write_enabled": False,
    "polymarket_write_enabled": False,
    "wallet_signing_enabled": False,
}


class PreflightError(ValueError):
    """Only fixed, sanitized reason codes may escape the validation boundary."""


def duration_seconds(value: str) -> float:
    """Strict positive integer s/m/h, at most seven days (no floats/exponents)."""
    match = re.fullmatch(r"([1-9][0-9]{0,6})([smh])", value)
    if match is None:
        raise ValueError("Duration must be a positive integer followed by s, m or h")
    seconds = int(match[1]) * {"s": 1, "m": 60, "h": 3600}[match[2]]
    if seconds > 7 * 86400:
        raise ValueError("Duration exceeds seven days")
    return float(seconds)


def validate_report_directory(report: Path, output: Path) -> None:
    report, output = report.expanduser().resolve(), output.expanduser().resolve()
    if report.is_relative_to(output) or output.is_relative_to(report):
        raise PreflightError("REPORT_OUTPUT_TREES_OVERLAP")
    if report.exists() and (not report.is_dir() or any(report.iterdir())):
        raise PreflightError("REPORT_DIRECTORY_NOT_EMPTY")


@dataclass(frozen=True)
class Preflight:
    config: OperatorConfig
    config_path: Path
    cwd: Path
    host: str
    port: int
    report_dir: Path | None
    mock: bool
    build_provenance: dict[str, str] | None = None

    @property
    def url(self) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"http://{host}:{self.port}"

    @property
    def identity(self) -> dict[str, str]:
        return {name: str(getattr(self.config, name)) for name in (
            "operator_id", "strategy_id", "paper_account_id", "source_commit", "config_sha256",
        )}

    @property
    def mode(self) -> str:
        return "mock_public_feeds_paper_execution" if self.mock else "live_public_feeds_paper_execution"

    def summary(self) -> dict[str, Any]:
        return {"valid": True, "paper_only": True, **self.identity,
                "dashboard_url": self.url, "mode": self.mode, "build_provenance": self.build_provenance}


def preflight(*, config_path: Path, host: str = "127.0.0.1", port: int = 8080,
              report_dir: Path | None = None, mock: bool = False) -> Preflight:
    try:
        config = load_operator_config(config_path)
        if any(getattr(config, key) is not value for key, value in SAFETY.items()):
            raise PreflightError("UNSAFE_CONFIGURATION")
        for name in ("operator_id", "strategy_id", "paper_account_id", "source_commit"):
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", str(getattr(config, name))):
                raise PreflightError("INVALID_PUBLIC_IDENTITY")
        if config.config_check_only:
            raise PreflightError("CONFIG_CHECK_ONLY_CANNOT_RUN")
        provenance = None
        if not mock:
            if config.mock is not False or config.dry_run is not False:
                raise PreflightError("LIVE_MODE_REQUIRES_MOCK_FALSE_DRY_RUN_FALSE")
            commit = config.source_commit
            if not re.fullmatch(r"[0-9a-f]{40}", commit):
                raise PreflightError("LIVE_REQUIRES_FULL_DEPLOYED_SOURCE_COMMIT")
            provenance = require_source_commit(commit)
        elif config.mock is not True:
            raise PreflightError("MOCK_MODE_REQUIRES_MOCK_TRUE")
        host = loopback_host(host)
        if type(port) is not int or not 1 <= port <= 65535:
            raise PreflightError("INVALID_DASHBOARD_PORT")
        if report_dir is not None:
            validate_report_directory(report_dir, Path(config.output_dir))
        with socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
            except OSError:
                raise PreflightError("DASHBOARD_PORT_UNAVAILABLE") from None
        return Preflight(config, config_path.resolve(), Path.cwd().resolve(), host, port,
                         None if report_dir is None else report_dir.expanduser().resolve(), mock, provenance)
    except BuildProvenanceError as exc:
        raise PreflightError(str(exc)) from None
    except PreflightError:
        raise
    except (OSError, ValueError, TypeError):
        raise PreflightError("INVALID_PAPER_STACK_CONFIGURATION") from None
