from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from bigan.execution import PositionManager

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "reconcile_stale_execution_positions.py"

spec = importlib.util.spec_from_file_location("reconcile_stale_execution_positions", SCRIPT)
assert spec is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)


def test_reconcile_stale_execution_positions_accepts_provider_settlement(
    tmp_path: Path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "positions.duckdb"
    csv_path = tmp_path / "history.csv"
    summary_path = tmp_path / "summary.json"
    event_id = "phase4-btc-updown-15m-1779786000-DOWN-loser"

    csv_path.write_text(
        "\n".join(
            [
                "marketName,action,usdcAmount,tokenAmount,tokenName,timestamp,hash",
                (
                    '"Bitcoin Up or Down - May 26, 5:00AM-5:15AM ET",'
                    "Buy,1.0,2.5,Down,1779786324,0xbuy"
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manager = PositionManager(db_path)
    manager.open_position(
        event_id,
        "BTC-15M:btc-updown-15m-1779786000:DOWN",
        "DOWN",
        0.4,
        2.5,
        "entry",
        fill_price=0.4,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--history-csv",
            str(csv_path),
            "--db-path",
            str(db_path),
            "--settlement-result",
            f"{event_id}=UP",
            "--summary-json-path",
            str(summary_path),
        ],
    )

    assert module.main() == 0

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["settled_from_provider"] == 1
    settled = manager.get_position(event_id)
    assert settled is not None
    assert settled.status == "expired"
    assert settled.settlement_result == "UP"
    assert settled.realized_pnl == -1.0
