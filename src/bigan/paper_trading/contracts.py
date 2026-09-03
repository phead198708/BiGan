"""Immutable, JSON-safe contracts for auditable paper trading runs."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from enum import Enum, StrEnum
from typing import Any

from bigan.pipeline.events import StrategyDecisionEvent

PAPER_SCHEMA_VERSION = "1.0"


class LedgerEventKind(StrEnum):
    """Kinds of state observations persisted by the paper ledger."""

    DECISION = "DECISION"
    MARK_TO_MARKET = "MARK_TO_MARKET"
    SETTLEMENT = "SETTLEMENT"


@dataclass(frozen=True, slots=True, kw_only=True)
class PaperSafetyBoundary:
    """Hard-coded proof that a contract cannot authorize a live write."""

    paper_only: bool = True
    capital_at_risk: bool = False
    broker_exchange_write_enabled: bool = False
    live_exchange_write_enabled: bool = False
    polymarket_write_enabled: bool = False
    wallet_signing_enabled: bool = False

    def __post_init__(self) -> None:
        if (
            self.paper_only is not True
            or self.capital_at_risk is not False
            or self.broker_exchange_write_enabled is not False
            or self.live_exchange_write_enabled is not False
            or self.polymarket_write_enabled is not False
            or self.wallet_signing_enabled is not False
        ):
            raise ValueError("paper-only safety boundary cannot be relaxed")

    def to_dict(self) -> dict[str, object]:
        return _to_dict(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class PaperWindowRegistration:
    """Window identity and expiry needed to validate settlement."""

    window_id: str
    market_symbol: str
    start_ts_ms: int
    end_ts_ms: int

    def __post_init__(self) -> None:
        _require_text("window_id", self.window_id)
        _require_text("market_symbol", self.market_symbol)
        if self.end_ts_ms <= self.start_ts_ms:
            raise ValueError("window end must be after window start")

    def to_dict(self) -> dict[str, object]:
        return _to_dict(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class PaperRunManifest(PaperSafetyBoundary):
    """Immutable identity and safety metadata for one paper run."""

    schema_version: str
    run_id: str
    created_at: str
    source_commit: str
    initial_bankroll: float
    fee_bps: float
    market_symbols: tuple[str, ...]
    window_ids: tuple[str, ...]
    windows: tuple[PaperWindowRegistration, ...]
    config_sha256: str

    def __post_init__(self) -> None:
        super(PaperRunManifest, self).__post_init__()
        _require_schema(self.schema_version)
        for name, value in (
            ("run_id", self.run_id),
            ("created_at", self.created_at),
            ("source_commit", self.source_commit),
            ("config_sha256", self.config_sha256),
        ):
            _require_text(name, value)
        _finite("initial_bankroll", self.initial_bankroll)
        _finite("fee_bps", self.fee_bps)
        if self.initial_bankroll <= 0.0:
            raise ValueError("initial_bankroll must be positive")
        if not 0.0 <= self.fee_bps <= 10_000.0:
            raise ValueError("fee_bps must be in [0, 10_000]")
        if not self.windows:
            raise ValueError("at least one window must be registered")
        expected_symbols = tuple(dict.fromkeys(row.market_symbol for row in self.windows))
        expected_ids = tuple(row.window_id for row in self.windows)
        if self.market_symbols != expected_symbols or self.window_ids != expected_ids:
            raise ValueError("manifest window identity does not match registrations")
        if len(set(self.window_ids)) != len(self.window_ids):
            raise ValueError("window ids must be unique")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PaperRunManifest:
        values = _exact_payload(cls, payload)
        values["market_symbols"] = tuple(values["market_symbols"])
        values["window_ids"] = tuple(values["window_ids"])
        values["windows"] = tuple(
            PaperWindowRegistration(**row) for row in values["windows"]
        )
        return cls(**values)


@dataclass(frozen=True, slots=True, kw_only=True)
class PaperDecisionEvent(PaperSafetyBoundary):
    """Run-scoped wrapper around one pipeline decision event."""

    schema_version: str
    run_id: str
    event_id: str
    event_sequence: int
    decision: StrategyDecisionEvent

    def __post_init__(self) -> None:
        super(PaperDecisionEvent, self).__post_init__()
        _validate_event_identity(self.schema_version, self.run_id, self.event_id, self.event_sequence)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PaperDecisionEvent:
        values = _exact_payload(cls, payload)
        values["decision"] = StrategyDecisionEvent.from_dict(values["decision"])
        return cls(**values)


@dataclass(frozen=True, slots=True, kw_only=True)
class PaperLot:
    """One immutable paper fill lot."""

    lot_id: str
    decision_event_id: str
    window_id: str
    market_symbol: str
    side: str
    shares: float
    entry_price: float
    entry_fee_usdc: float
    entry_ts_ms: int

    def __post_init__(self) -> None:
        for name, value in (
            ("lot_id", self.lot_id),
            ("decision_event_id", self.decision_event_id),
            ("window_id", self.window_id),
            ("market_symbol", self.market_symbol),
        ):
            _require_text(name, value)
        if self.side not in {"YES", "NO"}:
            raise ValueError("paper lot side must be YES or NO")
        for float_name, float_value in (
            ("shares", self.shares),
            ("entry_price", self.entry_price),
            ("entry_fee_usdc", self.entry_fee_usdc),
        ):
            _finite(float_name, float_value)
        if self.shares <= 0.0 or not 0.0 < self.entry_price <= 1.0:
            raise ValueError("lot shares and entry price are invalid")
        if self.entry_fee_usdc < 0.0:
            raise ValueError("entry fee cannot be negative")

    def to_dict(self) -> dict[str, object]:
        return _to_dict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PaperLot:
        return cls(**_exact_payload(cls, payload))


@dataclass(frozen=True, slots=True, kw_only=True)
class PaperPosition:
    """Aggregated open paper position for one window and side."""

    window_id: str
    market_symbol: str
    side: str
    shares: float
    average_entry_price: float
    cost_usdc: float
    entry_fees_usdc: float
    mark_bid: float
    market_value_usdc: float
    unrealized_pnl: float

    def __post_init__(self) -> None:
        _require_text("window_id", self.window_id)
        _require_text("market_symbol", self.market_symbol)
        if self.side not in {"YES", "NO"}:
            raise ValueError("paper position side must be YES or NO")
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, float):
                _finite(field.name, value)
        if self.shares <= 0.0 or self.cost_usdc <= 0.0:
            raise ValueError("position shares and cost must be positive")
        if not 0.0 <= self.mark_bid <= 1.0:
            raise ValueError("mark bid must be in [0, 1]")

    def to_dict(self) -> dict[str, object]:
        return _to_dict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PaperPosition:
        return cls(**_exact_payload(cls, payload))


@dataclass(frozen=True, slots=True, kw_only=True)
class PaperLedgerEvent(PaperSafetyBoundary):
    """Complete account observation after one deterministic ledger action."""

    schema_version: str
    run_id: str
    event_id: str
    event_sequence: int
    kind: LedgerEventKind
    source_event_id: str
    timestamp_ms: int
    window_id: str
    cash: float
    realized_pnl: float
    unrealized_pnl: float
    equity: float
    peak_equity: float
    drawdown: float
    commission_paid: float
    positions: tuple[PaperPosition, ...]
    open_lots: tuple[PaperLot, ...]

    def __post_init__(self) -> None:
        super(PaperLedgerEvent, self).__post_init__()
        _validate_event_identity(self.schema_version, self.run_id, self.event_id, self.event_sequence)
        _require_text("source_event_id", self.source_event_id)
        _require_text("window_id", self.window_id)
        _validate_account_values(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PaperLedgerEvent:
        values = _exact_payload(cls, payload)
        values["kind"] = LedgerEventKind(str(values["kind"]))
        values["positions"] = tuple(PaperPosition.from_dict(row) for row in values["positions"])
        values["open_lots"] = tuple(PaperLot.from_dict(row) for row in values["open_lots"])
        return cls(**values)


@dataclass(frozen=True, slots=True, kw_only=True)
class PaperSettlementInput:
    """Auditable external truth used to settle a binary market."""

    window_id: str
    yes_payout: float
    settlement_ts_ms: int
    source: str
    source_ts_ms: int
    received_ts_ms: int
    source_reference: str

    def __post_init__(self) -> None:
        for name, value in (
            ("window_id", self.window_id),
            ("source", self.source),
            ("source_reference", self.source_reference),
        ):
            _require_text(name, value)
        _finite("yes_payout", self.yes_payout)
        if not 0.0 <= self.yes_payout <= 1.0:
            raise ValueError("yes_payout must be in [0, 1]")
        if min(self.settlement_ts_ms, self.source_ts_ms, self.received_ts_ms) < 0:
            raise ValueError("settlement timestamps must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return _to_dict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PaperSettlementInput:
        return cls(**_exact_payload(cls, payload))


@dataclass(frozen=True, slots=True, kw_only=True)
class PaperSettlementEvent(PaperSafetyBoundary):
    """Persisted settlement mutation and resulting account values."""

    schema_version: str
    run_id: str
    event_id: str
    event_sequence: int
    settlement: PaperSettlementInput
    proceeds_usdc: float
    realized_pnl_delta: float
    cash_after: float
    realized_pnl: float
    commission_paid: float
    equity: float

    def __post_init__(self) -> None:
        super(PaperSettlementEvent, self).__post_init__()
        _validate_event_identity(self.schema_version, self.run_id, self.event_id, self.event_sequence)
        for name in (
            "proceeds_usdc",
            "realized_pnl_delta",
            "cash_after",
            "realized_pnl",
            "commission_paid",
            "equity",
        ):
            _finite(name, float(getattr(self, name)))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PaperSettlementEvent:
        values = _exact_payload(cls, payload)
        values["settlement"] = PaperSettlementInput.from_dict(values["settlement"])
        return cls(**values)


@dataclass(frozen=True, slots=True, kw_only=True)
class PaperAccountSnapshot(PaperSafetyBoundary):
    """Current recoverable paper account state."""

    schema_version: str
    run_id: str
    last_event_sequence: int
    timestamp_ms: int
    cash: float
    realized_pnl: float
    unrealized_pnl: float
    equity: float
    peak_equity: float
    drawdown: float
    commission_paid: float
    positions: tuple[PaperPosition, ...]
    open_lots: tuple[PaperLot, ...]
    settled_window_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        super(PaperAccountSnapshot, self).__post_init__()
        _require_schema(self.schema_version)
        _require_text("run_id", self.run_id)
        if self.last_event_sequence < 0:
            raise ValueError("last_event_sequence must be non-negative")
        _validate_account_values(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PaperAccountSnapshot:
        values = _exact_payload(cls, payload)
        values["positions"] = tuple(PaperPosition.from_dict(row) for row in values["positions"])
        values["open_lots"] = tuple(PaperLot.from_dict(row) for row in values["open_lots"])
        values["settled_window_ids"] = tuple(values["settled_window_ids"])
        return cls(**values)


def _validate_event_identity(
    schema_version: str,
    run_id: str,
    event_id: str,
    event_sequence: int,
) -> None:
    _require_schema(schema_version)
    _require_text("run_id", run_id)
    _require_text("event_id", event_id)
    if event_sequence <= 0:
        raise ValueError("event_sequence must be positive")


def _validate_account_values(value: PaperLedgerEvent | PaperAccountSnapshot) -> None:
    for name in (
        "cash",
        "realized_pnl",
        "unrealized_pnl",
        "equity",
        "peak_equity",
        "drawdown",
        "commission_paid",
    ):
        _finite(name, float(getattr(value, name)))
    if float(value.cash) < -1e-9:
        raise ValueError("cash cannot be negative")
    drawdown = float(value.drawdown)
    if not 0.0 <= drawdown <= 1.0:
        raise ValueError("drawdown must be in [0, 1]")


def _to_dict(value: Any) -> dict[str, object]:
    return {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def _exact_payload(cls: Any, payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != {field.name for field in fields(cls)}:
        raise ValueError(f"{cls.__name__} fields do not match schema")
    return dict(payload)


def _require_schema(value: str) -> None:
    if value != PAPER_SCHEMA_VERSION:
        raise ValueError("unsupported paper schema_version")


def _require_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _finite(name: str, value: float) -> None:
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
