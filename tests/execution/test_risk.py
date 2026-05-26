"""Risk manager tests for issue #74."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from bigan.execution import PositionManager, RiskConfig, RiskManager
from bigan.monitoring import open_data_quality_incidents


def _manager(tmp_path: Path, **config_overrides: object) -> RiskManager:
    positions = PositionManager(tmp_path / "positions.duckdb")
    return RiskManager(positions, config=RiskConfig(**config_overrides))


def test_kelly_sizing_for_expected_edges(tmp_path: Path) -> None:
    risk = _manager(tmp_path, max_concurrent_positions=10)

    edge_030 = risk.check_entry_allowed("edge-030", 0.30)
    edge_045 = risk.check_entry_allowed("edge-045", 0.45)
    edge_060 = risk.check_entry_allowed("edge-060", 0.60)

    assert edge_030.allowed is True
    assert edge_030.size == pytest.approx(3.0)
    assert edge_045.allowed is True
    assert edge_045.size == pytest.approx(4.5)
    assert edge_060.allowed is True
    assert edge_060.size == pytest.approx(10.0)


def test_concurrent_position_limit_blocks_new_entries(tmp_path: Path) -> None:
    positions = PositionManager(tmp_path / "positions.duckdb")
    positions.open_position("round-1", "symbol-1", "UP", 0.50, 1.0, "order-1")
    positions.open_position("round-2", "symbol-2", "DOWN", 0.50, 1.0, "order-2")
    risk = RiskManager(positions, config=RiskConfig(max_concurrent_positions=2))

    decision = risk.check_entry_allowed("round-3", 0.60)

    assert decision.allowed is False
    assert decision.reason == "max_concurrent_positions"


def test_daily_loss_circuit_breaker_records_incident(tmp_path: Path) -> None:
    conn = duckdb.connect()
    positions = PositionManager(tmp_path / "positions.duckdb")
    risk = RiskManager(
        positions,
        config=RiskConfig(daily_loss_limit_usdc=5.0, max_concurrent_positions=10),
        incident_conn=conn,
    )

    risk.record_loss(-5.0)
    decision = risk.check_entry_allowed("round-1", 0.60)

    assert decision.allowed is False
    assert decision.reason == "daily_loss_limit"
    incidents = open_data_quality_incidents(conn, severity="critical")
    assert len(incidents) == 1
    assert incidents[0]["alert_id"] == "CIRCUIT_BREAKER"


def test_min_order_size_blocks_tiny_edge(tmp_path: Path) -> None:
    risk = _manager(tmp_path, max_concurrent_positions=10)

    decision = risk.check_entry_allowed("tiny-edge", 0.01)

    assert decision.allowed is False
    assert decision.size == 0.0
    assert decision.reason == "size_below_min_order"
