"""Append-only JSONL storage and atomic snapshots for paper runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from bigan.pipeline.events import DecisionDisposition

from .contracts import (
    PaperAccountSnapshot,
    PaperDecisionEvent,
    PaperLedgerEvent,
    PaperRunManifest,
    PaperSettlementEvent,
)
from .ledger import PaperAccountLedger

MANIFEST_FILE = "paper_run_manifest.json"
SIGNAL_EVENTS_FILE = "signal_events.jsonl"
EXECUTION_EVENTS_FILE = "execution_events.jsonl"
LEDGER_EVENTS_FILE = "ledger_events.jsonl"
POSITION_SNAPSHOTS_FILE = "position_snapshots.jsonl"
PNL_SNAPSHOTS_FILE = "pnl_snapshots.jsonl"
SETTLEMENT_EVENTS_FILE = "settlement_events.jsonl"
SNAPSHOT_FILE = "paper_snapshot.json"
JSONL_FILES = (
    SIGNAL_EVENTS_FILE,
    EXECUTION_EVENTS_FILE,
    LEDGER_EVENTS_FILE,
    POSITION_SNAPSHOTS_FILE,
    PNL_SNAPSHOTS_FILE,
    SETTLEMENT_EVENTS_FILE,
)


class PaperRunStore:
    """Explicit new/resume lifecycle for a single durable paper run."""

    def __init__(self, *, run_dir: Path, manifest: PaperRunManifest, fsync: bool) -> None:
        self.run_dir = run_dir
        self.manifest = manifest
        self.fsync = fsync

    @classmethod
    def create_new(
        cls,
        *,
        output_dir: str | Path,
        manifest: PaperRunManifest,
        fsync: bool = False,
    ) -> PaperRunStore:
        """Create a run without ever overwriting an existing run directory."""

        _validate_run_component(manifest.run_id)
        run_dir = Path(output_dir).expanduser().resolve() / manifest.run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        store = cls(run_dir=run_dir, manifest=manifest, fsync=bool(fsync))
        store._write_atomic_json(MANIFEST_FILE, manifest.to_dict())
        for name in JSONL_FILES:
            (run_dir / name).touch(exist_ok=False)
        store._write_atomic_json(
            SNAPSHOT_FILE,
            PaperAccountLedger(
                run_id=manifest.run_id,
                initial_bankroll=manifest.initial_bankroll,
                windows=manifest.windows,
            ).snapshot().to_dict(),
        )
        return store

    @classmethod
    def load_manifest(
        cls,
        *,
        output_dir: str | Path,
        run_id: str,
    ) -> PaperRunManifest:
        """Load only the explicit run manifest; no directory discovery occurs."""

        _validate_run_component(run_id)
        run_dir = Path(output_dir).expanduser().resolve() / run_id
        if not run_dir.is_dir():
            raise FileNotFoundError(f"paper run does not exist: {run_dir}")
        return PaperRunManifest.from_dict(
            _load_json_object(run_dir / MANIFEST_FILE)
        )

    @classmethod
    def resume_existing(
        cls,
        *,
        output_dir: str | Path,
        expected_manifest: PaperRunManifest,
        fsync: bool = False,
    ) -> PaperRunStore:
        """Open a run only after strict manifest and artifact validation."""

        _validate_run_component(expected_manifest.run_id)
        run_dir = Path(output_dir).expanduser().resolve() / expected_manifest.run_id
        if not run_dir.is_dir():
            raise FileNotFoundError(f"paper run does not exist: {run_dir}")
        manifest_payload = _load_json_object(run_dir / MANIFEST_FILE)
        actual_manifest = PaperRunManifest.from_dict(manifest_payload)
        if actual_manifest != expected_manifest:
            raise ValueError("paper run manifest/config identity mismatch")
        store = cls(run_dir=run_dir, manifest=actual_manifest, fsync=bool(fsync))
        store.recover_ledger()
        return store

    def append_decision(
        self,
        *,
        decision: PaperDecisionEvent,
        ledger_event: PaperLedgerEvent,
        snapshot: PaperAccountSnapshot,
    ) -> None:
        """Append one strategy decision and its derived account observations."""

        self._validate_artifact_identity(
            decision.run_id,
            ledger_event.run_id,
            snapshot.run_id,
        )
        if (
            ledger_event.source_event_id != decision.event_id
            or ledger_event.event_sequence != decision.event_sequence
            or snapshot.last_event_sequence != decision.event_sequence
        ):
            raise ValueError("decision artifacts do not share one event identity")
        self._append_jsonl(SIGNAL_EVENTS_FILE, decision.to_dict())
        if decision.decision.disposition in {
            DecisionDisposition.NO_ORDER,
            DecisionDisposition.FILLED,
            DecisionDisposition.REJECTED,
        }:
            self._append_jsonl(EXECUTION_EVENTS_FILE, decision.to_dict())
        self._append_observation(ledger_event, snapshot)

    def load_decision_events(self) -> tuple[PaperDecisionEvent, ...]:
        """Load validated decisions for durable pre-OMS snapshot deduplication."""

        decisions = tuple(
            PaperDecisionEvent.from_dict(row)
            for row in self._read_jsonl(SIGNAL_EVENTS_FILE)
        )
        _validate_stream_order(list(decisions), SIGNAL_EVENTS_FILE)
        seen: dict[str, PaperDecisionEvent] = {}
        for event in decisions:
            existing = seen.get(event.source_snapshot_id)
            if existing is not None and existing != event:
                raise ValueError("conflicting duplicate source_snapshot_id")
            seen[event.source_snapshot_id] = event
        return decisions

    def append_settlement(
        self,
        *,
        settlement: PaperSettlementEvent,
        ledger_event: PaperLedgerEvent,
        snapshot: PaperAccountSnapshot,
    ) -> None:
        """Append settlement truth and its derived account observations."""

        self._validate_artifact_identity(
            settlement.run_id,
            ledger_event.run_id,
            snapshot.run_id,
        )
        if (
            ledger_event.source_event_id != settlement.event_id
            or ledger_event.event_sequence != settlement.event_sequence
            or snapshot.last_event_sequence != settlement.event_sequence
        ):
            raise ValueError("settlement artifacts do not share one event identity")
        self._append_jsonl(SETTLEMENT_EVENTS_FILE, settlement.to_dict())
        self._append_observation(ledger_event, snapshot)

    def recover_ledger(self) -> PaperAccountLedger:
        """Validate all artifacts, replay authoritative events, and verify snapshot."""

        decision_payloads = self._read_jsonl(SIGNAL_EVENTS_FILE)
        settlement_payloads = self._read_jsonl(SETTLEMENT_EVENTS_FILE)
        decisions = [PaperDecisionEvent.from_dict(row) for row in decision_payloads]
        settlements = [PaperSettlementEvent.from_dict(row) for row in settlement_payloads]
        _validate_stream_order(decisions, SIGNAL_EVENTS_FILE)
        _validate_stream_order(settlements, SETTLEMENT_EVENTS_FILE)
        authoritative: list[PaperDecisionEvent | PaperSettlementEvent] = [
            *decisions,
            *settlements,
        ]
        authoritative.sort(key=lambda event: event.event_sequence)

        ledger_payloads = self._read_jsonl(LEDGER_EVENTS_FILE)
        persisted_ledger = [PaperLedgerEvent.from_dict(row) for row in ledger_payloads]
        ledger = PaperAccountLedger(
            run_id=self.manifest.run_id,
            initial_bankroll=self.manifest.initial_bankroll,
            windows=self.manifest.windows,
        )
        generated_ledger: list[PaperLedgerEvent] = []
        generated_snapshots: list[PaperAccountSnapshot] = []
        for event in authoritative:
            if isinstance(event, PaperDecisionEvent):
                generated = ledger.apply_decision(event)
                if generated is None:
                    continue
            else:
                sequence_before = ledger.last_event_sequence
                actual = ledger.settle(
                    event.settlement,
                    event_id=event.event_id,
                    event_sequence=event.event_sequence,
                )
                if actual != event:
                    raise ValueError("settlement result differs during replay")
                if ledger.last_event_sequence == sequence_before:
                    continue
                generated = ledger.settlement_ledger_event(actual)
            generated_ledger.append(generated)
            generated_snapshots.append(ledger.snapshot())
        if generated_ledger != persisted_ledger:
            raise ValueError("persisted ledger events differ from deterministic replay")

        execution_payloads = self._read_jsonl(EXECUTION_EVENTS_FILE)
        executions = [PaperDecisionEvent.from_dict(row) for row in execution_payloads]
        expected_executions = [
            event
            for event in decisions
            if event.decision.disposition
            in {
                DecisionDisposition.NO_ORDER,
                DecisionDisposition.FILLED,
                DecisionDisposition.REJECTED,
            }
        ]
        if executions != expected_executions:
            raise ValueError("execution events do not match strategy decisions")

        position_rows = [
            PaperAccountSnapshot.from_dict(row)
            for row in self._read_jsonl(POSITION_SNAPSHOTS_FILE)
        ]
        pnl_rows = [
            PaperAccountSnapshot.from_dict(row)
            for row in self._read_jsonl(PNL_SNAPSHOTS_FILE)
        ]
        if position_rows != generated_snapshots or position_rows != pnl_rows:
            raise ValueError("position/PnL snapshot streams are inconsistent")
        current = PaperAccountSnapshot.from_dict(
            _load_json_object(self.run_dir / SNAPSHOT_FILE)
        )
        recovered = ledger.snapshot()
        if current != recovered or (position_rows and position_rows[-1] != recovered):
            raise ValueError("current paper snapshot differs from event replay")
        if not position_rows and recovered.last_event_sequence != 0:
            raise ValueError("missing snapshot observations for non-empty run")
        return ledger

    def _append_observation(
        self,
        ledger_event: PaperLedgerEvent,
        snapshot: PaperAccountSnapshot,
    ) -> None:
        self._append_jsonl(LEDGER_EVENTS_FILE, ledger_event.to_dict())
        self._append_jsonl(POSITION_SNAPSHOTS_FILE, snapshot.to_dict())
        self._append_jsonl(PNL_SNAPSHOTS_FILE, snapshot.to_dict())
        self._write_atomic_json(SNAPSHOT_FILE, snapshot.to_dict())

    def _validate_artifact_identity(self, *run_ids: str) -> None:
        if any(run_id != self.manifest.run_id for run_id in run_ids):
            raise ValueError("artifact run_id does not match paper run manifest")

    def _append_jsonl(self, name: str, payload: dict[str, object]) -> None:
        encoded = _encode_json(payload) + b"\n"
        path = self.run_dir / name
        with path.open("ab") as handle:
            handle.write(encoded)
            handle.flush()
            if self.fsync:
                os.fsync(handle.fileno())

    def _read_jsonl(self, name: str) -> list[dict[str, Any]]:
        path = self.run_dir / name
        raw = path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            raise ValueError(f"truncated JSONL file: {name}")
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(raw.splitlines(), start=1):
            try:
                value = json.loads(line, parse_constant=_reject_json_constant)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"invalid JSONL at {name}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row must be an object at {name}:{line_number}")
            rows.append(value)
        return rows

    def _write_atomic_json(self, name: str, payload: dict[str, object]) -> None:
        path = self.run_dir / name
        temporary = self.run_dir / f".{name}.{os.getpid()}.tmp"
        encoded = _encode_json(payload) + b"\n"
        try:
            with temporary.open("xb") as handle:
                handle.write(encoded)
                handle.flush()
                if self.fsync:
                    os.fsync(handle.fileno())
            os.replace(temporary, path)
            if self.fsync:
                directory_fd = os.open(self.run_dir, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)


def _validate_run_component(run_id: str) -> None:
    if run_id in {".", ".."} or Path(run_id).name != run_id:
        raise ValueError("run_id must be a single safe path component")


def _encode_json(payload: dict[str, object]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes(), parse_constant=_reject_json_constant)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid JSON object: {path.name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path.name}")
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _validate_stream_order(events: list[Any], name: str) -> None:
    previous = 0
    for event in events:
        if event.event_sequence < previous:
            raise ValueError(f"event_sequence moves backwards in {name}")
        previous = event.event_sequence
