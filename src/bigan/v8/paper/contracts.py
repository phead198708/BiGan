"""Paper-only execution contracts for the v8 trading architecture."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

PAPER_TRADING_HARNESS_PHASE = "paper_trading_harness"
DEFAULT_PAPER_CREATED_AT = "1970-01-01T00:00:00Z"
PRIMARY_PAPER_ARTIFACT_FILENAMES: tuple[str, ...] = (
    "paper_orders.jsonl",
    "paper_fills.jsonl",
    "paper_ledger.jsonl",
    "paper_positions.json",
    "paper_pnl_report.json",
    "paper_bundle_manifest.json",
)
PAPER_ARTIFACT_FILENAMES: tuple[str, ...] = (
    *PRIMARY_PAPER_ARTIFACT_FILENAMES,
    "phase5_safety_layer_report.json",
    "phase6_cicd_pipeline_report_<release_id>.json",
)

PaperSide = Literal["buy", "sell", "hold"]


class PaperTradingError(RuntimeError):
    """Raised when the paper harness receives unsafe inputs."""


@dataclass(frozen=True, slots=True)
class PaperDegradationConfig:
    """Deterministic paper degradation injection for safety-layer tests."""

    start_index: int
    net_return_shift: float = 0.0
    cost_multiplier: float = 1.0
    live_regime: str | None = None

    def __post_init__(self) -> None:
        if self.start_index < 0:
            raise ValueError("start_index must be non-negative")
        if not math.isfinite(self.net_return_shift):
            raise ValueError("net_return_shift must be finite")
        if self.cost_multiplier <= 0.0 or not math.isfinite(self.cost_multiplier):
            raise ValueError("cost_multiplier must be positive and finite")
        if self.live_regime is not None and not self.live_regime.strip():
            raise ValueError("live_regime must be non-empty when provided")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PaperHarnessConfig:
    """Configuration and identity evidence for one paper-only harness run."""

    run_id: str
    candidate_run_id: str
    model_sha256: str
    policy_dataset_hash: str
    split_hash: str
    output_dir: Path | str | None = None
    created_at: str = DEFAULT_PAPER_CREATED_AT
    initial_cash: float = 100_000.0
    base_mark_price: float = 100.0
    min_fill_probability: float = 0.0
    degradation: PaperDegradationConfig | None = None
    upstream_training_report_sha256: str = "2" * 64
    upstream_validation_report_sha256: str = "3" * 64
    overwrite_existing: bool = False
    broker_write_enabled: bool = False
    paper_only: bool = True
    capital_at_risk: bool = False

    def __post_init__(self) -> None:
        if self.output_dir is not None and not isinstance(self.output_dir, Path):
            object.__setattr__(self, "output_dir", Path(self.output_dir))
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if not self.candidate_run_id.strip():
            raise ValueError("candidate_run_id is required")
        for field_name in (
            "model_sha256",
            "policy_dataset_hash",
            "split_hash",
            "upstream_training_report_sha256",
            "upstream_validation_report_sha256",
        ):
            if not looks_like_sha256(str(getattr(self, field_name))):
                raise ValueError(f"{field_name} must be a SHA-256 hex digest")
        if self.initial_cash <= 0.0 or not math.isfinite(self.initial_cash):
            raise ValueError("initial_cash must be positive and finite")
        if self.base_mark_price <= 0.0 or not math.isfinite(self.base_mark_price):
            raise ValueError("base_mark_price must be positive and finite")
        if not 0.0 <= self.min_fill_probability <= 1.0:
            raise ValueError("min_fill_probability must be in [0, 1]")
        if self.broker_write_enabled:
            raise ValueError("paper harness cannot enable broker/exchange writes")
        if self.paper_only is not True:
            raise ValueError("paper_only must be true")
        if self.capital_at_risk is not False:
            raise ValueError("capital_at_risk must be false")
        if not self.created_at:
            raise ValueError("created_at is required")

    def identity_metadata(self) -> dict[str, str]:
        return {
            "candidate_run_id": self.candidate_run_id,
            "model_sha256": self.model_sha256,
            "policy_dataset_hash": self.policy_dataset_hash,
            "split_hash": self.split_hash,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_dir"] = None if self.output_dir is None else str(self.output_dir)
        payload["degradation"] = (
            None if self.degradation is None else self.degradation.to_dict()
        )
        return payload


@dataclass(frozen=True, slots=True)
class PaperOrder:
    """Paper-only target-position order derived from one Phase 4 decision."""

    order_id: str
    candidate_run_id: str
    decision_ts: int
    source: str
    instrument_id: str
    side: PaperSide
    previous_action: float
    requested_action: float
    requested_size: float
    limit_price: float
    order_type: str
    created_at_ts: int
    paper_only: bool = True
    capital_at_risk: bool = False
    metadata: Mapping[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        _validate_common_paper_fields(self)
        if self.side not in ("buy", "sell", "hold"):
            raise ValueError("side must be buy, sell, or hold")
        for field_name in (
            "previous_action",
            "requested_action",
            "requested_size",
            "limit_price",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
        if not 0.0 <= self.previous_action <= 1.0:
            raise ValueError("previous_action must be in [0, 1]")
        if not 0.0 <= self.requested_action <= 1.0:
            raise ValueError("requested_action must be in [0, 1]")
        if self.requested_size < 0.0:
            raise ValueError("requested_size must be non-negative")
        if self.limit_price <= 0.0:
            raise ValueError("limit_price must be positive")
        if not self.order_type:
            raise ValueError("order_type is required")
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True, slots=True)
class PaperFill:
    """Deterministic paper fill generated from a paper order."""

    fill_id: str
    order_id: str
    decision_ts: int
    source: str
    instrument_id: str
    side: PaperSide
    requested_size: float
    filled_size: float
    filled_action: float
    fill_price: float
    mark_price: float
    fill_probability: float
    spread_cost: float
    fee_cost: float
    slippage_cost: float
    liquidity_impact_cost: float
    total_execution_cost: float
    net_return: float
    paper_regime: str
    paper_only: bool = True
    capital_at_risk: bool = False
    metadata: Mapping[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        _validate_common_paper_fields(self)
        if not self.fill_id:
            raise ValueError("fill_id is required")
        if not self.order_id:
            raise ValueError("order_id is required")
        if self.side not in ("buy", "sell", "hold"):
            raise ValueError("side must be buy, sell, or hold")
        for field_name in (
            "requested_size",
            "filled_size",
            "filled_action",
            "fill_price",
            "mark_price",
            "fill_probability",
            "spread_cost",
            "fee_cost",
            "slippage_cost",
            "liquidity_impact_cost",
            "total_execution_cost",
            "net_return",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
        if self.requested_size < 0.0:
            raise ValueError("requested_size must be non-negative")
        if self.filled_size < 0.0:
            raise ValueError("filled_size must be non-negative")
        if not 0.0 <= self.filled_action <= 1.0:
            raise ValueError("filled_action must be in [0, 1]")
        if self.fill_price <= 0.0 or self.mark_price <= 0.0:
            raise ValueError("fill_price and mark_price must be positive")
        if not 0.0 <= self.fill_probability <= 1.0:
            raise ValueError("fill_probability must be in [0, 1]")
        expected_cost = (
            self.spread_cost
            + self.fee_cost
            + self.slippage_cost
            + self.liquidity_impact_cost
        )
        if abs(self.total_execution_cost - expected_cost) > 1e-12:
            raise ValueError("total_execution_cost must equal component costs")
        if self.total_execution_cost < 0.0:
            raise ValueError("total_execution_cost must be non-negative")
        if not self.paper_regime:
            raise ValueError("paper_regime is required")
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True, slots=True)
class PaperLedgerEntry:
    """Paper-only ledger entry produced by applying a fill.

    Accounting units are synthetic and deterministic: position fields are
    normalized action fractions, cash fields are fixture notional balances, and
    ``net_return`` / ``realized_pnl`` are normalized return values rather than
    real-currency PnL.
    """

    entry_id: str
    order_id: str
    fill_id: str
    decision_ts: int
    source: str
    instrument_id: str
    position_before: float
    position_after: float
    cash_before: float
    cash_after: float
    realized_pnl: float
    unrealized_pnl: float
    net_return: float
    cumulative_net_return: float
    total_execution_cost: float
    paper_only: bool = True
    capital_at_risk: bool = False

    def __post_init__(self) -> None:
        _validate_common_paper_fields(self)
        for field_name in (
            "position_before",
            "position_after",
            "cash_before",
            "cash_after",
            "realized_pnl",
            "unrealized_pnl",
            "net_return",
            "cumulative_net_return",
            "total_execution_cost",
        ):
            if not math.isfinite(float(getattr(self, field_name))):
                raise ValueError(f"{field_name} must be finite")
        if self.total_execution_cost < 0.0:
            raise ValueError("total_execution_cost must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PaperPositionSnapshot:
    """Final deterministic paper position state for one instrument."""

    source: str
    instrument_id: str
    position_size: float
    average_entry_price: float
    mark_price: float
    unrealized_pnl: float
    realized_pnl: float
    last_update_ts: int
    paper_only: bool = True
    capital_at_risk: bool = False

    def __post_init__(self) -> None:
        _validate_common_paper_fields(self)
        for field_name in (
            "position_size",
            "average_entry_price",
            "mark_price",
            "unrealized_pnl",
            "realized_pnl",
        ):
            if not math.isfinite(float(getattr(self, field_name))):
                raise ValueError(f"{field_name} must be finite")
        if self.average_entry_price < 0.0:
            raise ValueError("average_entry_price must be non-negative")
        if self.mark_price <= 0.0:
            raise ValueError("mark_price must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PaperRunReport:
    """Auditable immutable paper harness report used as Phase 6 input evidence.

    The Phase 6 report hash is intentionally recorded in the bundle manifest
    only, keeping ``paper_pnl_report.json`` stable after Phase 6 consumes it.
    """

    phase: str
    run_id: str
    candidate_run_id: str
    model_sha256: str
    policy_dataset_hash: str
    split_hash: str
    paper_only: bool
    capital_at_risk: bool
    row_count: int
    order_count: int
    fill_count: int
    ledger_entry_count: int
    paper_order_stream_sha256: str
    paper_fill_stream_sha256: str
    paper_ledger_sha256: str
    paper_positions_sha256: str
    mean_net_return: float
    max_drawdown: float
    total_execution_cost: float
    phase5_report_sha256: str | None
    acceptance_criteria: Mapping[str, bool]
    config: Mapping[str, Any]
    created_at: str

    @property
    def passed(self) -> bool:
        return all(self.acceptance_criteria.values())

    def __post_init__(self) -> None:
        if self.phase != PAPER_TRADING_HARNESS_PHASE:
            raise ValueError(f"phase must be {PAPER_TRADING_HARNESS_PHASE!r}")
        if self.paper_only is not True:
            raise ValueError("paper_only must be true")
        if self.capital_at_risk is not False:
            raise ValueError("capital_at_risk must be false")
        for field_name in (
            "model_sha256",
            "policy_dataset_hash",
            "split_hash",
            "paper_order_stream_sha256",
            "paper_fill_stream_sha256",
            "paper_ledger_sha256",
            "paper_positions_sha256",
        ):
            if not looks_like_sha256(str(getattr(self, field_name))):
                raise ValueError(f"{field_name} must be a SHA-256 hex digest")
        if self.phase5_report_sha256 is not None and not looks_like_sha256(
            self.phase5_report_sha256
        ):
            raise ValueError("phase5_report_sha256 must be a SHA-256 hex digest")
        if not self.acceptance_criteria:
            raise ValueError("acceptance_criteria must not be empty")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["acceptance_criteria"] = dict(self.acceptance_criteria)
        payload["config"] = dict(self.config)
        payload["passed"] = self.passed
        return payload


def canonical_payload_sha256(payload: Any) -> str:
    """Hash a JSON-serializable payload with deterministic formatting."""

    encoded = json.dumps(
        json_ready(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stream_sha256(records: tuple[Any, ...] | list[Any]) -> str:
    """Hash an ordered stream of dataclass-like records."""

    return canonical_payload_sha256(
        [record.to_dict() if hasattr(record, "to_dict") else record for record in records]
    )


def json_ready(value: Any) -> Any:
    """Convert dataclass payloads and pathlib values into JSON-safe values."""

    if hasattr(value, "to_dict"):
        return json_ready(value.to_dict())
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    return value


def looks_like_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value.lower())


def _validate_common_paper_fields(record: Any) -> None:
    if not record.source:
        raise ValueError("source is required")
    if not record.instrument_id:
        raise ValueError("instrument_id is required")
    if int(getattr(record, "decision_ts", getattr(record, "last_update_ts", 0))) < 0:
        raise ValueError("timestamp must be non-negative")
    if record.paper_only is not True:
        raise ValueError("paper_only must be true")
    if record.capital_at_risk is not False:
        raise ValueError("capital_at_risk must be false")
