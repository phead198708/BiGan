"""Account cash-flow reconciliation tests for issue #79."""

from __future__ import annotations

import duckdb
import pytest

from bigan.execution import (
    PositionManager,
    account_cash_pnl,
    read_cashflow_reconciliations,
    read_polymarket_history_csv,
    reconcile_cash_flows,
    record_cashflow_reconciliations,
)


def test_cashflow_reconciliation_uses_account_cash_and_tracks_dust(tmp_path) -> None:
    csv_path = tmp_path / "history.csv"
    csv_path.write_text(
        "\n".join(
            [
                "marketName,action,usdcAmount,tokenAmount,tokenName,timestamp,hash",
                (
                    '"Bitcoin Up or Down - May 26, 5:00AM-5:15AM ET",'
                    "Sell,0.33214,3.22,Up,1779786634,0xsell"
                ),
                (
                    '"Bitcoin Up or Down - May 26, 5:00AM-5:15AM ET",'
                    "Buy,1.048289,3.225805,Up,1779786324,0xbuy"
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    positions = PositionManager(tmp_path / "positions.duckdb")
    positions.open_position(
        "phase4-btc-updown-15m-1779786000-UP-10f86d62",
        "BTC-15M:btc-updown-15m-1779786000:UP",
        "UP",
        0.31,
        3.225805,
        "order-entry",
        fill_price=0.31,
        sleeve="volatility",
    )
    positions.close_position(
        "phase4-btc-updown-15m-1779786000-UP-10f86d62",
        0.11,
        exit_time=1779786634000,
    )

    flows = read_polymarket_history_csv(csv_path)
    records = reconcile_cash_flows(positions.list_positions(), flows)

    assert flows[0].round_slug == "btc-updown-15m-1779786000"
    assert account_cash_pnl(flows) == pytest.approx(-0.716149)
    assert len(records) == 1
    record = records[0]
    assert record.sleeve == "volatility"
    assert record.match_status == "matched"
    assert record.account_cash_pnl == pytest.approx(-0.716149)
    assert record.theoretical_pnl == pytest.approx(-0.645161)
    assert record.cash_pnl_delta == pytest.approx(-0.070988)
    assert record.dust_token_amount == pytest.approx(0.005805)


def test_redeem_cash_flow_is_assigned_to_single_round_position(tmp_path) -> None:
    csv_path = tmp_path / "history.csv"
    csv_path.write_text(
        "\n".join(
            [
                "marketName,action,usdcAmount,tokenAmount,tokenName,timestamp,hash",
                (
                    '"Bitcoin Up or Down - May 26, 1:45AM-2:00AM ET",'
                    "Redeem,2.127658,2.127658,,1779775240,0xredeem"
                ),
                (
                    '"Bitcoin Up or Down - May 26, 1:45AM-2:00AM ET",'
                    "Buy,1.037089,2.127658,Up,1779773992,0xbuy"
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    positions = PositionManager(tmp_path / "positions.duckdb")
    positions.open_position(
        "phase4-btc-updown-15m-1779774300-UP-a7fc2f63",
        "BTC-15M:btc-updown-15m-1779774300:UP",
        "UP",
        0.47,
        2.127658,
        "order-entry",
        fill_price=0.47,
    )

    records = reconcile_cash_flows(positions.list_positions(), read_polymarket_history_csv(csv_path))

    assert len(records) == 1
    record = records[0]
    assert record.round_slug == "btc-updown-15m-1779774300"
    assert record.match_status == "matched"
    assert record.account_cash_pnl == pytest.approx(1.090569)
    assert record.redeemed_token_amount == pytest.approx(2.127658)
    assert record.dust_token_amount == pytest.approx(0.0)
    assert record.cash_pnl_delta is None


def test_redeem_is_not_ambitiously_assigned_when_round_has_two_positions(tmp_path) -> None:
    csv_path = tmp_path / "history.csv"
    csv_path.write_text(
        "\n".join(
            [
                "marketName,action,usdcAmount,tokenAmount,tokenName,timestamp,hash",
                (
                    '"Bitcoin Up or Down - May 26, 1:15AM-1:30AM ET",'
                    "Redeem,2.0,2.0,,1779773440,0xredeem"
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    positions = PositionManager(tmp_path / "positions.duckdb")
    positions.open_position("phase4-btc-updown-15m-1779772500-UP-a", "s", "UP", 0.5, 2, "up")
    positions.open_position("phase4-btc-updown-15m-1779772500-DOWN-b", "s", "DOWN", 0.5, 2, "down")

    records = reconcile_cash_flows(positions.list_positions(), read_polymarket_history_csv(csv_path))

    assert {record.match_status for record in records} == {"ambiguous_redeem"}
    assert all(record.account_cash_pnl is None for record in records)


def test_cashflow_reconciliations_are_persisted(tmp_path) -> None:
    positions = PositionManager(tmp_path / "positions.duckdb")
    positions.open_position("phase4-btc-updown-15m-1779786000-UP-x", "s", "UP", 0.5, 1, "order")
    records = reconcile_cash_flows(positions.list_positions(), [])
    conn = duckdb.connect()

    record_cashflow_reconciliations(conn, records)
    rows = read_cashflow_reconciliations(conn)

    assert len(rows) == 1
    assert rows[0]["event_id"] == "phase4-btc-updown-15m-1779786000-UP-x"
    assert rows[0]["sleeve"] == "settlement"
    assert rows[0]["match_status"] == "missing_cash_flow"


def test_existing_cashflow_table_without_sleeve_is_migrated() -> None:
    conn = duckdb.connect()
    conn.execute(
        """
        CREATE TABLE execution_cashflow_reconciliations (
            event_id VARCHAR PRIMARY KEY,
            round_slug VARCHAR NOT NULL,
            side VARCHAR NOT NULL,
            position_status VARCHAR NOT NULL,
            theoretical_pnl DOUBLE,
            account_cash_pnl DOUBLE,
            cash_pnl_delta DOUBLE,
            bought_token_amount DOUBLE NOT NULL,
            sold_token_amount DOUBLE NOT NULL,
            redeemed_token_amount DOUBLE NOT NULL,
            dust_token_amount DOUBLE NOT NULL,
            cash_flow_count INTEGER NOT NULL,
            first_cash_flow_ts BIGINT,
            last_cash_flow_ts BIGINT,
            match_status VARCHAR NOT NULL,
            cash_flows_json VARCHAR NOT NULL,
            created_at BIGINT NOT NULL,
            updated_at BIGINT NOT NULL
        )
        """
    )
    conn.execute(
        """
        INSERT INTO execution_cashflow_reconciliations VALUES (
            'event-1', 'btc-updown-15m-1', 'UP', 'open', NULL, NULL, NULL,
            0.0, 0.0, 0.0, 0.0, 0, NULL, NULL, 'missing_cash_flow',
            '[]', 1000, 1000
        )
        """
    )

    rows = read_cashflow_reconciliations(conn)

    assert rows[0]["sleeve"] == "settlement"
