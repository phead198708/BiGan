"""One local command; live means public data, never real money."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .observer import ObservationPolicy
from .preflight import PreflightError, duration_seconds, preflight
from .supervisor import PaperStackSupervisor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PAPER / SIMULATED — NO REAL FUNDS: local paper stack")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dashboard-host", default="127.0.0.1")
    parser.add_argument("--dashboard-port", type=int, default=8080)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--no-soak-report", action="store_true", help="Keep observation gates; do not write artifacts")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--mock-demo", action="store_true")
    parser.add_argument("--duration", type=duration_seconds)
    parser.add_argument("--startup-timeout", type=duration_seconds, default=60)
    parser.add_argument("--poll-interval", type=duration_seconds, default=2)
    parser.add_argument("--shutdown-grace", type=duration_seconds, default=15)
    parser.add_argument("--request-timeout", type=duration_seconds, default=3)
    parser.add_argument("--unreadable-timeout", type=duration_seconds, default=30)
    parser.add_argument("--stale-timeout", type=duration_seconds, default=30)
    parser.add_argument("--rollover-timeout", type=duration_seconds, default=900)
    args = parser.parse_args(argv)
    if args.no_soak_report and args.report_dir is not None:
        parser.error("Choose --report-dir or --no-soak-report, not both")
    if not args.preflight and not args.no_soak_report and args.report_dir is None:
        parser.error("An explicit --report-dir is required unless --no-soak-report is selected")
    try:
        check = preflight(config_path=args.config, host=args.dashboard_host, port=args.dashboard_port,
                          report_dir=args.report_dir, mock=args.mock_demo)
    except PreflightError as exc:
        print("[soak] preflight failed: " + str(exc))
        return 2
    if args.preflight:
        print(json.dumps(check.summary(), allow_nan=False, sort_keys=True))
        return 0
    policy = ObservationPolicy(request_timeout=args.request_timeout, unreadable_seconds=args.unreadable_timeout,
                               stale_seconds=args.stale_timeout, rollover_seconds=args.rollover_timeout)
    try:
        return asyncio.run(PaperStackSupervisor(
            check, duration=args.duration, startup_timeout=args.startup_timeout, poll_interval=args.poll_interval,
            shutdown_grace=args.shutdown_grace, policy=policy,
        ).run())
    except Exception:
        print("[soak] STACK_FAILURE (details suppressed for privacy)")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
