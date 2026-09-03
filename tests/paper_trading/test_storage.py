"""Append-only storage, atomic snapshot, and strict recovery tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from bigan.paper_trading.contracts import PaperSettlementInput
from bigan.paper_trading.ledger import PaperAccountLedger
from bigan.paper_trading.storage import (
    EXECUTION_EVENTS_FILE,
    JSONL_FILES,
    LEDGER_EVENTS_FILE,
    MANIFEST_FILE,
    PNL_SNAPSHOTS_FILE,
    POSITION_SNAPSHOTS_FILE,
    SIGNAL_EVENTS_FILE,
    SNAPSHOT_FILE,
    PaperRunStore,
)
from tests.paper_trading.helpers import RUN_ID, WINDOW, manifest, paper_decision


def _created_store(tmp_path: Path) -> tuple[PaperRunStore, PaperAccountLedger]:
    store = PaperRunStore.create_new(output_dir=tmp_path, manifest=manifest())
    ledger = PaperAccountLedger(
        run_id=RUN_ID,
        initial_bankroll=1_000.0,
        windows=(WINDOW,),
    )
    return store, ledger


def _append_fill(store: PaperRunStore, ledger: PaperAccountLedger) -> None:
    decision = paper_decision(1)
    event = ledger.apply_decision(decision)
    assert event is not None
    store.append_decision(
        decision=decision,
        ledger_event=event,
        snapshot=ledger.snapshot(),
    )


def test_create_new_layout_append_and_atomic_snapshot(tmp_path: Path) -> None:
    store, ledger = _created_store(tmp_path)
    assert (store.run_dir / MANIFEST_FILE).is_file()
    assert all((store.run_dir / name).is_file() for name in JSONL_FILES)
    _append_fill(store, ledger)

    signal_rows = [
        json.loads(line) for line in (store.run_dir / SIGNAL_EVENTS_FILE).read_text().splitlines()
    ]
    assert signal_rows == [paper_decision(1).to_dict()]
    assert json.loads((store.run_dir / SNAPSHOT_FILE).read_text()) == ledger.snapshot().to_dict()
    assert not list(store.run_dir.glob("*.tmp"))
    assert not list(store.run_dir.glob(".*.tmp"))
    for name in (LEDGER_EVENTS_FILE, POSITION_SNAPSHOTS_FILE, PNL_SNAPSHOTS_FILE):
        payload = json.loads((store.run_dir / name).read_text().splitlines()[0])
        assert payload["paper_only"] is True
        assert payload["live_exchange_write_enabled"] is False


def test_create_refuses_overwrite_and_resume_requires_existing(tmp_path: Path) -> None:
    _created_store(tmp_path)
    with pytest.raises(FileExistsError):
        PaperRunStore.create_new(output_dir=tmp_path, manifest=manifest())
    with pytest.raises(FileNotFoundError):
        PaperRunStore.resume_existing(
            output_dir=tmp_path,
            expected_manifest=replace(manifest(), run_id="missing"),
        )


def test_resume_replays_exact_state_and_sequence(tmp_path: Path) -> None:
    store, ledger = _created_store(tmp_path)
    _append_fill(store, ledger)
    expected = ledger.snapshot()

    resumed = PaperRunStore.resume_existing(
        output_dir=tmp_path,
        expected_manifest=manifest(),
    )
    recovered = resumed.recover_ledger()
    assert recovered.snapshot() == expected
    second = paper_decision(2, cash_before=expected.cash)
    mutation = recovered.apply_decision(second)
    assert mutation is not None
    assert recovered.last_event_sequence == 2


def test_storage_recovers_settlement(tmp_path: Path) -> None:
    store, ledger = _created_store(tmp_path)
    _append_fill(store, ledger)
    settlement_input = PaperSettlementInput(
        window_id=WINDOW.window_id,
        yes_payout=1.0,
        settlement_ts_ms=WINDOW.end_ts_ms,
        source="fixture",
        source_ts_ms=WINDOW.end_ts_ms,
        received_ts_ms=WINDOW.end_ts_ms + 1,
        source_reference="settlement-1",
    )
    settlement = ledger.settle(
        settlement_input,
        event_id="settlement-2",
        event_sequence=2,
    )
    store.append_settlement(
        settlement=settlement,
        ledger_event=ledger.settlement_ledger_event(settlement),
        snapshot=ledger.snapshot(),
    )

    recovered = PaperRunStore.resume_existing(
        output_dir=tmp_path,
        expected_manifest=manifest(),
    ).recover_ledger()
    assert recovered.snapshot() == ledger.snapshot()


def test_manifest_config_mismatch_fails(tmp_path: Path) -> None:
    _created_store(tmp_path)
    with pytest.raises(ValueError, match="identity mismatch"):
        PaperRunStore.resume_existing(
            output_dir=tmp_path,
            expected_manifest=manifest(config_sha256="b" * 64),
        )


@pytest.mark.parametrize("damage", [b'{"broken":}\n', b'{"truncated":true}'])
def test_malformed_or_truncated_jsonl_fails_closed(tmp_path: Path, damage: bytes) -> None:
    store, _ledger = _created_store(tmp_path)
    (store.run_dir / SIGNAL_EVENTS_FILE).write_bytes(damage)
    with pytest.raises(ValueError, match="JSONL"):
        PaperRunStore.resume_existing(
            output_dir=tmp_path,
            expected_manifest=manifest(),
        )


def test_valid_but_inconsistent_derived_log_fails_closed(tmp_path: Path) -> None:
    store, ledger = _created_store(tmp_path)
    _append_fill(store, ledger)
    (store.run_dir / LEDGER_EVENTS_FILE).write_text("")
    with pytest.raises(ValueError, match="persisted ledger"):
        PaperRunStore.resume_existing(
            output_dir=tmp_path,
            expected_manifest=manifest(),
        )


def test_exact_duplicate_authoritative_event_replays_idempotently(tmp_path: Path) -> None:
    store, ledger = _created_store(tmp_path)
    _append_fill(store, ledger)
    encoded = json.dumps(
        paper_decision(1).to_dict(),
        sort_keys=True,
        separators=(",", ":"),
    )
    for name in (SIGNAL_EVENTS_FILE, EXECUTION_EVENTS_FILE):
        with (store.run_dir / name).open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")

    recovered = PaperRunStore.resume_existing(
        output_dir=tmp_path,
        expected_manifest=manifest(),
    ).recover_ledger()
    assert recovered.snapshot() == ledger.snapshot()


def test_conflicting_duplicate_authoritative_event_fails(tmp_path: Path) -> None:
    store, ledger = _created_store(tmp_path)
    _append_fill(store, ledger)
    original = paper_decision(1)
    conflicting_decision = replace(original.decision, yes_bid=0.20)
    conflicting = replace(
        original,
        decision=conflicting_decision,
        source_snapshot_id=conflicting_decision.source_snapshot_id,
    )
    with (store.run_dir / SIGNAL_EVENTS_FILE).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(conflicting.to_dict()) + "\n")
    with pytest.raises(ValueError, match="conflicting duplicate"):
        PaperRunStore.resume_existing(
            output_dir=tmp_path,
            expected_manifest=manifest(),
        )


def test_non_finite_json_and_unsafe_manifest_fail_closed(tmp_path: Path) -> None:
    store, _ledger = _created_store(tmp_path)
    manifest_path = store.run_dir / MANIFEST_FILE
    payload = manifest().to_dict()
    payload["initial_bankroll"] = float("nan")
    manifest_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="invalid JSON"):
        PaperRunStore.resume_existing(
            output_dir=tmp_path,
            expected_manifest=manifest(),
        )

    payload = manifest().to_dict()
    payload["paper_only"] = False
    manifest_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="safety boundary"):
        PaperRunStore.resume_existing(
            output_dir=tmp_path,
            expected_manifest=manifest(),
        )


def test_run_id_cannot_escape_explicit_output_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="path component"):
        PaperRunStore.load_manifest(output_dir=tmp_path, run_id="../elsewhere")
