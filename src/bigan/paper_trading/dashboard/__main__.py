"""Run only the local dashboard, never the trading operator."""

from __future__ import annotations

import argparse

from aiohttp import web

from bigan.paper_trading.operator.config import load_operator_config

from .reader import DashboardReader
from .server import create_app, loopback_host


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local read-only paper trading dashboard")
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8080, type=int)
    args = parser.parse_args(argv)
    try:
        host = loopback_host(args.host)
        if not 1 <= args.port <= 65535:
            raise ValueError("invalid port")
        config = load_operator_config(args.config)
    except (OSError, ValueError, TypeError):
        parser.error("Invalid operator configuration, loopback host, or port")
    try:
        web.run_app(create_app(DashboardReader(config)), host=host, port=args.port, access_log=None)
    except OSError:
        parser.error("Unable to start local dashboard listener")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
