from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import duckdb

from bigan.execution import PositionManager, read_cashflow_reconciliations

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "reconcile_polymarket_cashflows.py"

spec = importlib.util.spec_from_file_location("reconcile_polymarket_cashflows", SCRIPT)
assert spec is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)


def test_reconcile_polymarket_cashflows_writes_report_and_table(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "positions.duckdb"
    csv_path = tmp_path / "history.csv"
    report_path = tmp_path / "report.md"
    summary_path = tmp_path / "summary.json"
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
    positions = PositionManager(db_path)
    positions.open_position(
        "phase4-btc-updown-15m-1779786000-UP-10f86d62",
        "BTC-15M:btc-updown-15m-1779786000:UP",
        "UP",
        0.31,
        3.225805,
        "order-entry",
        fill_price=0.31,
    )
    positions.close_position("phase4-btc-updown-15m-1779786000-UP-10f86d62", 0.11)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--history-csv",
            str(csv_path),
            "--db-path",
            str(db_path),
            "--write-db",
            "--report-path",
            str(report_path),
            "--summary-json-path",
            str(summary_path),
        ],
    )

    assert module.main() == 0

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["matched"] == 1
    assert summary["account_cash_pnl"] == -0.716149
    assert summary["by_sleeve"]["settlement"]["positions"] == 1
    assert summary["by_sleeve"]["settlement"]["account_cash_pnl"] == -0.716149
    assert "Account cash-flow PnL: -0.716149" in report_path.read_text(encoding="utf-8")
    assert "| settlement | 1 | 1 | 0 | -0.716149 |" in report_path.read_text(encoding="utf-8")
    conn = duckdb.connect(str(db_path))
    rows = read_cashflow_reconciliations(conn)
    assert rows[0]["sleeve"] == "settlement"
    assert rows[0]["cash_pnl_delta"] < 0


def test_reconcile_polymarket_cashflows_filters_explicit_event_ids(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "positions.duckdb"
    csv_path = tmp_path / "history.csv"
    summary_path = tmp_path / "summary.json"
    selected_event_id = "phase4-btc-updown-15m-1779786000-UP-selected"
    historical_event_id = "phase4-btc-updown-15m-1779786900-DOWN-history"
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
                (
                    '"Bitcoin Up or Down - May 26, 5:15AM-5:30AM ET",'
                    "Sell,24.0,24.0,Down,1779787520,0xold-sell"
                ),
                (
                    '"Bitcoin Up or Down - May 26, 5:15AM-5:30AM ET",'
                    "Buy,1.0,2.0,Down,1779787000,0xold-buy"
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    positions = PositionManager(db_path)
    positions.open_position(
        selected_event_id,
        "BTC-15M:btc-updown-15m-1779786000:UP",
        "UP",
        0.31,
        3.225805,
        "order-entry",
        fill_price=0.31,
    )
    positions.close_position(selected_event_id, 0.11)
    positions.open_position(
        historical_event_id,
        "BTC-15M:btc-updown-15m-1779786900:DOWN",
        "DOWN",
        0.50,
        2.0,
        "old-entry",
        fill_price=0.50,
    )
    positions.close_position(historical_event_id, 1.0)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--history-csv",
            str(csv_path),
            "--db-path",
            str(db_path),
            "--event-id",
            selected_event_id,
            "--summary-json-path",
            str(summary_path),
        ],
    )

    assert module.main() == 0

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["positions"] == 1
    assert summary["matched"] == 1
    assert summary["account_cash_pnl"] == -0.716149
    assert set(summary["by_sleeve"]) == {"settlement"}
