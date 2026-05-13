"""Shared pytest configuration."""

from __future__ import annotations

import os
from typing import Any

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="Run tests marked 'live' against public upstream APIs.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[Any]) -> None:
    marker_expr = getattr(config.option, "markexpr", "") or ""
    live_requested = (
        config.getoption("--run-live")
        or os.getenv("BIGAN_RUN_LIVE_TESTS") == "1"
        or "live" in marker_expr
    )
    if live_requested:
        return

    skip_live = pytest.mark.skip(
        reason="set BIGAN_RUN_LIVE_TESTS=1, pass --run-live, or select -m live"
    )
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
