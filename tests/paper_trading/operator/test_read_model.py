from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from bigan.paper_trading.contracts import PaperSettlementInput
from bigan.paper_trading.ledger import PaperAccountLedger
from bigan.paper_trading.operator.read_model import (
    OPERATOR_STATUS_SCHEMA_VERSION,
    OperatorReadRepository,
    OperatorState,
    OperatorStatus,
    OperatorStatusWriter,
)
from bigan.paper_trading.storage import PaperRunStore
from bigan.pipeline.events import DecisionDisposition
from tests.paper_trading.helpers import RUN_ID, WINDOW, manifest, paper_decision


def _status(*, state: OperatorState = OperatorState.RUNNING) -> OperatorStatus:
    return OperatorStatus(
        schema_version=OPERATOR_STATUS_SCHEMA_VERSION,
        operator_id="operator-a",
        strategy_id="strategy-a",
        run_id=RUN_ID,
        state=state,
        state_reason="all_sources_fresh",
        process_started_at_ms=1_000,
        updated_at_ms=2_000,
        source_commit="deadbeef",
        paper_only=True,
        safety={
            "capital_at_risk": False,
            "exchange_write_enabled": False,
            "wallet_signing_enabled": False,
        },
        active_market={
            "market_id": "market-a",
            "window_id": WINDOW.window_id,
            "yes_token_id": "yes",
            "no_token_id": "no",
            "start_ts_ms": WINDOW.start_ts_ms,
            "end_ts_ms": WINDOW.end_ts_ms,
            "seconds_to_end": 8,
        },
        feeds={"binance": {"state": "READY"}, "polymarket": {"state": "READY"}},
        pricing_inputs={"fresh": True, "timestamp_ms": 2_000, "age_ms": 0},
        alpha={"fresh": True, "timestamp_ms": 2_000, "age_ms": 0, "z_score": 1.2},
        session={"healthy": True, "failure_reason": None},
        account={
            "initial_bankroll": 1_000.0,
            "cash": 996.0,
            "equity": 1_000.0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "total_fees": 0.0,
            "open_positions": [],
        },
        counters={"decisions": 1, "fills": 1, "rejects": 0, "holds": 0, "drops": 0},
        last_decision=None,
        last_fill=None,
        settlement={"status": "OPEN", "source_reference": None},
    )


def _store_with_history(tmp_path: Path) -> tuple[PaperRunStore, PaperAccountLedger]:
    store = PaperRunStore.create_new(output_dir=tmp_path, manifest=manifest())
    ledger = PaperAccountLedger(run_id=RUN_ID, initial_bankroll=1_000.0, windows=(WINDOW,))
    cash = 1_000.0
    for sequence, disposition in enumerate(
        (
            DecisionDisposition.FILLED,
            DecisionDisposition.REJECTED,
            DecisionDisposition.FILLED,
        ),
        start=1,
    ):
        event = paper_decision(sequence, disposition=disposition, cash_before=cash)
        mutation = ledger.apply_decision(event)
        assert mutation is not None
        cash = ledger.cash
        store.append_decision(decision=event, ledger_event=mutation, snapshot=ledger.snapshot())
    return store, ledger


def test_status_atomic_round_trip_and_required_dashboard_fields(tmp_path: Path) -> None:
    path = tmp_path / "operator_status.json"
    writer = OperatorStatusWriter(path)
    writer.write(_status())

    loaded = OperatorReadRepository(status_path=path, run_store=None).current_status()
    assert loaded == _status()
    assert loaded.account["open_positions"] == []
    assert loaded.settlement["status"] == "OPEN"
    assert not list(tmp_path.glob(".*.tmp"))


def test_non_finite_status_rejected_without_replacing_previous(tmp_path: Path) -> None:
    path = tmp_path / "operator_status.json"
    writer = OperatorStatusWriter(path)
    writer.write(_status())
    original = path.read_bytes()

    with pytest.raises(ValueError, match="NaN"):
        replace(_status(), account={**_status().account, "equity": float("nan")})
    assert path.read_bytes() == original


def test_atomic_replace_failure_preserves_previous_projection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "operator_status.json"
    writer = OperatorStatusWriter(path)
    writer.write(_status())
    original = path.read_bytes()

    def fail_replace(_source: object, _destination: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="disk full"):
        writer.write(replace(_status(), state_reason="newer_projection"))
    assert path.read_bytes() == original


def test_recent_queries_are_bounded_and_filter_fills(tmp_path: Path) -> None:
    store, _ledger = _store_with_history(tmp_path)
    path = tmp_path / "status.json"
    OperatorStatusWriter(path).write(_status())
    repository = OperatorReadRepository(
        status_path=path,
        run_store=store,
        default_limit=2,
        max_limit=2,
    )

    assert [row["event_sequence"] for row in repository.recent_decisions()] == [2, 3]
    assert [row["event_sequence"] for row in repository.recent_fills()] == [1, 3]
    with pytest.raises(ValueError, match="bounds"):
        repository.recent_decisions(3)


def test_settlement_query_and_settled_status_contract(tmp_path: Path) -> None:
    store, ledger = _store_with_history(tmp_path)
    settlement = ledger.settle(
        PaperSettlementInput(
            window_id=WINDOW.window_id,
            yes_payout=1.0,
            settlement_ts_ms=WINDOW.end_ts_ms,
            source="fixture",
            source_ts_ms=WINDOW.end_ts_ms,
            received_ts_ms=WINDOW.end_ts_ms + 1,
            source_reference="resolution-1",
        ),
        event_id="settlement-4",
        event_sequence=4,
    )
    store.append_settlement(
        settlement=settlement,
        ledger_event=ledger.settlement_ledger_event(settlement),
        snapshot=ledger.snapshot(),
    )
    path = tmp_path / "status.json"
    settled = replace(
        _status(),
        state=OperatorState.ROLLING_OVER,
        state_reason="settlement_persisted",
        account={**_status().account, "realized_pnl": ledger.realized_pnl},
        settlement={"status": "SETTLED", "source_reference": "resolution-1"},
    )
    OperatorStatusWriter(path).write(settled)
    repository = OperatorReadRepository(status_path=path, run_store=store)

    assert repository.current_status().settlement["status"] == "SETTLED"
    assert repository.settlements()[0]["settlement"]["source_reference"] == "resolution-1"


def test_corrupt_projection_does_not_damage_authoritative_recovery(tmp_path: Path) -> None:
    store, ledger = _store_with_history(tmp_path)
    path = tmp_path / "status.json"
    path.write_text('{"equity":NaN}\n')
    repository = OperatorReadRepository(status_path=path, run_store=store)

    with pytest.raises(ValueError, match="projection"):
        repository.current_status()
    assert store.recover_ledger().snapshot() == ledger.snapshot()
    assert repository.current_account() == ledger.snapshot()


def test_status_schema_rejects_unknown_fields_and_failed_state_is_complete(tmp_path: Path) -> None:
    failed = replace(
        _status(),
        state=OperatorState.FAILED,
        state_reason="persistence_failure",
        session={"healthy": False, "failure_reason": "disk full"},
    )
    payload = failed.to_dict()
    payload["secret"] = "must never be tolerated"
    with pytest.raises(ValueError, match="schema"):
        OperatorStatus.from_dict(payload)

    path = tmp_path / "status.json"
    OperatorStatusWriter(path).write(failed)
    raw = json.loads(path.read_text())
    assert raw["state"] == "FAILED"
    assert raw["session"]["failure_reason"] == "disk full"


def test_status_with_open_position_keeps_complete_account_shape(tmp_path: Path) -> None:
    account = {
        **_status().account,
        "cash": 960.0,
        "equity": 1_001.0,
        "unrealized_pnl": 1.0,
        "open_positions": [
            {
                "window_id": WINDOW.window_id,
                "side": "YES",
                "shares": 100.0,
                "market_value_usdc": 41.0,
            }
        ],
    }
    path = tmp_path / "operator_status.json"
    OperatorStatusWriter(path).write(replace(_status(), account=account))
    loaded = OperatorReadRepository(status_path=path, run_store=None).current_status()
    assert set(loaded.account) == {
        "initial_bankroll",
        "cash",
        "equity",
        "realized_pnl",
        "unrealized_pnl",
        "total_fees",
        "open_positions",
    }
    assert loaded.account["open_positions"] == account["open_positions"]
