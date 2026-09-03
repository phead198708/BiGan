"""Atomic, derived operator projection and bounded read-only queries."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from bigan.paper_trading.contracts import PaperAccountSnapshot
from bigan.paper_trading.storage import SNAPSHOT_FILE, PaperRunStore

OPERATOR_STATUS_SCHEMA_VERSION = "1.0"


class OperatorState(StrEnum):
    STARTING = "STARTING"
    DISCOVERING = "DISCOVERING"
    SYNCING = "SYNCING"
    RUNNING = "RUNNING"
    DEGRADED = "DEGRADED"
    SETTLEMENT_PENDING = "SETTLEMENT_PENDING"
    ROLLING_OVER = "ROLLING_OVER"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class OperatorStatus:
    """Stable schema consumed by the future dashboard."""

    schema_version: str
    operator_id: str
    strategy_id: str
    run_id: str | None
    state: OperatorState
    state_reason: str
    process_started_at_ms: int
    updated_at_ms: int
    source_commit: str
    paper_only: bool
    safety: dict[str, object]
    active_market: dict[str, object] | None
    feeds: dict[str, object]
    pricing_inputs: dict[str, object]
    alpha: dict[str, object]
    session: dict[str, object]
    account: dict[str, object]
    counters: dict[str, int]
    last_decision: dict[str, object] | None
    last_fill: dict[str, object] | None
    settlement: dict[str, object]

    def __post_init__(self) -> None:
        if self.schema_version != OPERATOR_STATUS_SCHEMA_VERSION:
            raise ValueError("unsupported operator status schema")
        for name in ("operator_id", "strategy_id", "state_reason", "source_commit"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        if self.run_id is not None and not self.run_id.strip():
            raise ValueError("run_id must be null or non-empty")
        if self.paper_only is not True:
            raise ValueError("operator status must remain paper-only")
        if self.process_started_at_ms < 0 or self.updated_at_ms < self.process_started_at_ms:
            raise ValueError("operator status timestamps are invalid")
        for name in (
            "safety",
            "feeds",
            "pricing_inputs",
            "alpha",
            "session",
            "account",
            "counters",
            "settlement",
        ):
            if not isinstance(getattr(self, name), dict):
                raise ValueError(f"{name} must be an object")
        for name in ("active_market", "last_decision", "last_fill"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, dict):
                raise ValueError(f"{name} must be null or an object")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in self.counters.values()
        ):
            raise ValueError("operator counters must be non-negative integers")
        _require_finite_json(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["state"] = self.state.value
        return payload

    @classmethod
    def from_dict(cls, payload: object) -> OperatorStatus:
        if not isinstance(payload, dict):
            raise ValueError("operator status must be an object")
        expected = set(cls.__dataclass_fields__)
        if set(payload) != expected:
            raise ValueError("operator status fields do not match schema")
        values = dict(payload)
        values["state"] = OperatorState(str(values["state"]))
        return cls(**values)


class OperatorStatusWriter:
    """Write projection snapshots with temp-file + atomic replace."""

    def __init__(self, path: str | Path, *, fsync: bool = False) -> None:
        self.path = Path(path)
        self.fsync = bool(fsync)

    def write(self, status: OperatorStatus) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = _encode_json(status.to_dict()) + b"\n"
        temporary = self.path.parent / f".{self.path.name}.{os.getpid()}.tmp"
        temporary.unlink(missing_ok=True)
        try:
            with temporary.open("xb") as handle:
                handle.write(encoded)
                handle.flush()
                if self.fsync:
                    os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            if self.fsync:
                directory_fd = os.open(self.path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)


class OperatorReadRepository:
    """Pure read layer; callers never need to understand ledger internals."""

    def __init__(
        self,
        *,
        status_path: str | Path,
        run_store: PaperRunStore | None,
        default_limit: int = 50,
        max_limit: int = 500,
    ) -> None:
        if default_limit < 1 or max_limit < 1 or default_limit > max_limit:
            raise ValueError("read query bounds are invalid")
        self.status_path = Path(status_path)
        self.run_store = run_store
        self.default_limit = int(default_limit)
        self.max_limit = int(max_limit)

    def current_status(self) -> OperatorStatus:
        try:
            payload = json.loads(
                self.status_path.read_bytes(),
                parse_constant=_reject_constant,
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError("operator status projection is invalid") from exc
        return OperatorStatus.from_dict(payload)

    def current_account(self) -> PaperAccountSnapshot | None:
        if self.run_store is None:
            return None
        try:
            payload = json.loads(
                (self.run_store.run_dir / SNAPSHOT_FILE).read_bytes(),
                parse_constant=_reject_constant,
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError("paper account projection is invalid") from exc
        if not isinstance(payload, dict):
            raise ValueError("paper account projection must be an object")
        return PaperAccountSnapshot.from_dict(payload)

    def recent_decisions(self, limit: int | None = None) -> tuple[dict[str, object], ...]:
        store = self._require_store()
        bounded = self._limit(limit)
        return tuple(row.to_dict() for row in store.recent_decisions(limit=bounded, max_limit=self.max_limit))

    def recent_fills(self, limit: int | None = None) -> tuple[dict[str, object], ...]:
        store = self._require_store()
        bounded = self._limit(limit)
        return tuple(row.to_dict() for row in store.recent_fills(limit=bounded, max_limit=self.max_limit))

    def settlements(self, limit: int | None = None) -> tuple[dict[str, object], ...]:
        store = self._require_store()
        bounded = self._limit(limit)
        return tuple(row.to_dict() for row in store.recent_settlements(limit=bounded, max_limit=self.max_limit))

    def _limit(self, value: int | None) -> int:
        limit = self.default_limit if value is None else value
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= self.max_limit:
            raise ValueError("query limit is outside configured bounds")
        return limit

    def _require_store(self) -> PaperRunStore:
        if self.run_store is None:
            raise ValueError("there is no active paper run")
        return self.run_store


def _encode_json(payload: dict[str, object]) -> bytes:
    _require_finite_json(payload)
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _require_finite_json(value: object) -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("operator status cannot contain NaN or Infinity")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("operator status keys must be strings")
            _require_finite_json(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _require_finite_json(item)
    elif value is not None and not isinstance(value, (str, int, bool)):
        raise ValueError("operator status contains a non-JSON value")


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")
