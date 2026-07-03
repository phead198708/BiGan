"""Position manager tests for issue #73."""

from __future__ import annotations

import duckdb
import pytest

from bigan.execution import PositionManager


def test_existing_position_table_without_sleeve_is_migrated(tmp_path) -> None:
    db_path = tmp_path / "positions.duckdb"

    with duckdb.connect(str(db_path)) as conn:
        conn.execute(
            """
            CREATE TABLE execution_positions (
                event_id VARCHAR PRIMARY KEY,
                symbol VARCHAR NOT NULL,
                side VARCHAR NOT NULL,
                status VARCHAR NOT NULL,
                entry_time BIGINT NOT NULL,
                entry_price DOUBLE NOT NULL,
                fill_price DOUBLE,
                size DOUBLE NOT NULL,
                order_id VARCHAR NOT NULL,
                current_price DOUBLE,
                unrealized_pnl DOUBLE,
                exit_price DOUBLE,
                exit_time BIGINT,
                realized_pnl DOUBLE,
                settlement_result VARCHAR,
                created_at BIGINT NOT NULL,
                updated_at BIGINT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO execution_positions VALUES (
                'old-round', 'BTC-15M:old:UP', 'UP', 'open', 1000, 0.5,
                NULL, 2.0, 'order-1', 0.5, 0.0, NULL, NULL, NULL, NULL,
                1000, 1000
            )
            """
        )

    manager = PositionManager(db_path)

    migrated = manager.get_position("old-round")
    assert migrated is not None
    assert migrated.sleeve == "settlement"


def test_position_manager_retries_transient_duckdb_lock(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "positions.duckdb"
    real_connect = duckdb.connect
    calls = 0
    monkeypatch.setenv("BIGAN_EXECUTION_DB_CONNECT_RETRY_DELAY_SECONDS", "0")

    def flaky_connect(path: str, *args: object, **kwargs: object):
        nonlocal calls
        if path == str(db_path):
            calls += 1
            if calls == 1:
                raise duckdb.IOException(
                    "IO Error: Could not set lock on file positions.duckdb"
                )
        return real_connect(path, *args, **kwargs)

    monkeypatch.setattr(duckdb, "connect", flaky_connect)

    manager = PositionManager(db_path)

    assert calls >= 2
    assert manager.get_all_open() == []


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
    assert opened.sleeve == "settlement"
    assert manager.has_open_position("round-1") is True

    restarted = PositionManager(db_path)
    open_positions = restarted.get_all_open()
    assert len(open_positions) == 1
    assert open_positions[0].sleeve == "settlement"

    updated = restarted.update_price("round-1", 0.62)
    assert updated.unrealized_pnl == pytest.approx(0.22)

    adjusted = restarted.adjust_open_position(
        "round-1",
        fill_price=0.55,
        size=3.0,
        current_price=0.60,
    )
    assert adjusted.fill_price == pytest.approx(0.55)
    assert adjusted.size == pytest.approx(3.0)
    assert adjusted.current_price == pytest.approx(0.60)
    assert adjusted.unrealized_pnl == pytest.approx(0.15)

    closed = restarted.close_position("round-1", 0.64, exit_time=2000)
    assert closed.status == "closed"
    assert closed.realized_pnl == pytest.approx(0.27)
    assert restarted.has_open_position("round-1") is False


def test_position_sleeve_is_persisted(tmp_path) -> None:
    manager = PositionManager(tmp_path / "positions.duckdb")

    opened = manager.open_position(
        "vol-round-1",
        "BTC-15M:btc-updown-15m-1:UP",
        "UP",
        0.42,
        2.0,
        "order-1",
        sleeve="volatility",
    )

    restarted = PositionManager(tmp_path / "positions.duckdb")
    assert opened.sleeve == "volatility"
    assert restarted.get_position("vol-round-1").sleeve == "volatility"


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
