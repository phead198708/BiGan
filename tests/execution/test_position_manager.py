"""Position manager tests for issue #73."""

from __future__ import annotations

import pytest

from bigan.execution import PositionManager


def test_position_lifecycle_is_persisted_to_duckdb(tmp_path) -> None:
    db_path = tmp_path / "positions.duckdb"
    manager = PositionManager(db_path)

    opened = manager.open_position(
        "round-1",
        "BTC-15M:btc-updown-15m-1:UP",
        "UP",
        0.51,
        2.0,
        "order-1",
        entry_time=1000,
    )
    assert opened.status == "open"
    assert manager.has_open_position("round-1") is True

    restarted = PositionManager(db_path)
    assert len(restarted.get_all_open()) == 1

    updated = restarted.update_price("round-1", 0.62)
    assert updated.unrealized_pnl == pytest.approx(0.22)

    closed = restarted.close_position("round-1", 0.64, exit_time=2000)
    assert closed.status == "closed"
    assert closed.realized_pnl == pytest.approx(0.26)
    assert restarted.has_open_position("round-1") is False


def test_duplicate_open_position_is_rejected(tmp_path) -> None:
    manager = PositionManager(tmp_path / "positions.duckdb")
    manager.open_position("round-1", "symbol", "DOWN", 0.50, 1.0, "order-1")

    with pytest.raises(ValueError, match="already exists"):
        manager.open_position("round-1", "symbol", "DOWN", 0.50, 1.0, "order-2")


def test_settle_position_marks_winner_and_loser(tmp_path) -> None:
    manager = PositionManager(tmp_path / "positions.duckdb")
    manager.open_position("up-round", "symbol-up", "UP", 0.40, 3.0, "order-up")
    manager.open_position("down-round", "symbol-down", "DOWN", 0.55, 2.0, "order-down")

    winner = manager.settle_position("up-round", "UP", settlement_time=3000)
    loser = manager.settle_position("down-round", True, settlement_time=3000)

    assert winner.status == "expired"
    assert winner.exit_price == 1.0
    assert winner.realized_pnl == pytest.approx(1.8)
    assert loser.status == "expired"
    assert loser.exit_price == 0.0
    assert loser.realized_pnl == pytest.approx(-1.1)
