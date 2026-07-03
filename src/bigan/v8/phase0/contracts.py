"""Strict Phase 0 data contracts.

The v8 Phase 0 layer is a data-correctness firewall. These contracts make the
time semantics explicit:

* ``ts`` is the market event timestamp.
* ``available_at_ts`` is the earliest decision timestamp at which the row may
  be consumed by a feature.
* feature rows carry per-column provenance so causality can be checked
  mechanically.
* labels are the only contract that may point into the future.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PHASE0_DATASET_VERSION = "bigan-v8-phase0-v1.0.0"
FEATURE_VERSION = "bigan-v8-phase0-causal-features-v1.0.0"
LABEL_VERSION = "bigan-v8-phase0-cost-aware-labels-v1.0.0"

FEATURE_COLUMNS: tuple[str, ...] = (
    "mid_price",
    "spread",
    "spread_bps",
    "return_1m",
    "return_5m",
    "return_15m",
    "volatility_5m",
    "volatility_15m",
    "volume_1m",
    "volume_5m",
    "trade_count_1m",
    "trade_count_5m",
    "orderbook_imbalance_l1",
    "liquidity_depth",
    "minute_of_day",
    "day_of_week",
)

COST_COLUMNS: tuple[str, ...] = (
    "spread_cost",
    "fee_cost",
    "slippage_cost",
    "liquidity_impact_cost",
    "total_cost",
    "net_return",
)


class MarketData(BaseModel):
    """One normalized market observation.

    The loader accepts source-specific aliases, but every downstream Phase 0
    component consumes this normalized contract.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    ts: int = Field(ge=0)
    available_at_ts: int | None = Field(default=None, ge=0)
    source: str = Field(min_length=1)
    instrument_id: str = Field(min_length=1)
    bid_price: float | None = Field(default=None, gt=0)
    ask_price: float | None = Field(default=None, gt=0)
    mid_price: float | None = Field(default=None, gt=0)
    last_price: float | None = Field(default=None, gt=0)
    volume: float = Field(default=0.0, ge=0)
    trade_count: int = Field(default=0, ge=0)
    bid_size: float | None = Field(default=None, ge=0)
    ask_size: float | None = Field(default=None, ge=0)
    liquidity_depth: float | None = Field(default=None, ge=0)
    timeframe_ms: int | None = Field(default=None, gt=0)
    sequence: int | None = Field(default=None, ge=0)

    @model_validator(mode="before")
    @classmethod
    def derive_defaults(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        row = dict(data)
        if row.get("available_at_ts") is None:
            row["available_at_ts"] = row.get("ts")
        bid = _optional_float(row.get("bid_price"))
        ask = _optional_float(row.get("ask_price"))
        if row.get("mid_price") is None and bid is not None and ask is not None:
            row["mid_price"] = (bid + ask) / 2.0
        if row.get("liquidity_depth") is None:
            bid_size = _optional_float(row.get("bid_size"))
            ask_size = _optional_float(row.get("ask_size"))
            if bid_size is not None or ask_size is not None:
                row["liquidity_depth"] = (bid_size or 0.0) + (ask_size or 0.0)
        return row

    @field_validator("source", "instrument_id")
    @classmethod
    def strip_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must be non-empty")
        return stripped

    @model_validator(mode="after")
    def validate_prices_and_time(self) -> MarketData:
        if self.available_at_ts is None:
            raise ValueError("available_at_ts is required after defaulting")
        if self.available_at_ts < self.ts:
            raise ValueError("available_at_ts cannot be earlier than ts")
        if (
            self.bid_price is not None
            and self.ask_price is not None
            and self.bid_price > self.ask_price
        ):
            raise ValueError("bid_price cannot exceed ask_price")
        if self.mid_price is None and self.last_price is None:
            raise ValueError("mid_price, bid/ask pair, or last_price is required")
        return self

    @property
    def effective_mid_price(self) -> float:
        if self.mid_price is not None:
            return self.mid_price
        if self.bid_price is not None and self.ask_price is not None:
            return (self.bid_price + self.ask_price) / 2.0
        if self.last_price is not None:
            return self.last_price
        raise ValueError("market row has no usable price")

    def to_row(self) -> dict[str, Any]:
        return self.model_dump()


class FeatureProvenance(BaseModel):
    """Point-in-time provenance for one feature column."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    feature_name: str = Field(min_length=1)
    input_start_ts: int = Field(ge=0)
    input_end_ts: int = Field(ge=0)
    available_at_ts: int = Field(ge=0)
    lookback_ms: int = Field(ge=0)
    source_timeframe_ms: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_window(self) -> FeatureProvenance:
        if self.input_start_ts > self.input_end_ts:
            raise ValueError("input_start_ts cannot exceed input_end_ts")
        if self.available_at_ts < self.input_end_ts:
            raise ValueError("available_at_ts cannot be earlier than input_end_ts")
        return self


class FeatureVector(BaseModel):
    """Causal feature row at one decision timestamp."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    decision_ts: int = Field(ge=0)
    feature_cutoff_ts: int = Field(ge=0)
    lookback_start_ts: int = Field(ge=0)
    max_input_ts: int = Field(ge=0)
    source: str = Field(min_length=1)
    instrument_id: str = Field(min_length=1)
    feature_version: str = Field(default=FEATURE_VERSION, min_length=1)
    features: dict[str, float | int | None]
    provenance: dict[str, FeatureProvenance]

    @model_validator(mode="after")
    def validate_causality(self) -> FeatureVector:
        if self.feature_cutoff_ts > self.decision_ts:
            raise ValueError("feature_cutoff_ts cannot exceed decision_ts")
        if self.max_input_ts > self.decision_ts:
            raise ValueError("max_input_ts cannot exceed decision_ts")
        missing_provenance = set(self.features) - set(self.provenance)
        if missing_provenance:
            raise ValueError(
                "every feature must have provenance; missing "
                + ", ".join(sorted(missing_provenance))
            )
        return self

    def flat_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "decision_ts": self.decision_ts,
            "feature_cutoff_ts": self.feature_cutoff_ts,
            "lookback_start_ts": self.lookback_start_ts,
            "max_input_ts": self.max_input_ts,
            "source": self.source,
            "instrument_id": self.instrument_id,
            "feature_version": self.feature_version,
        }
        for column in FEATURE_COLUMNS:
            row[column] = self.features.get(column)
        return row


class Label(BaseModel):
    """Cost-aware future label.

    Labels are allowed to reference future data. This contract still requires
    all future access to be explicit via ``label_ts`` and ``horizon_ms``.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    decision_ts: int = Field(ge=0)
    label_ts: int = Field(ge=0)
    horizon_ms: int = Field(gt=0)
    source: str = Field(min_length=1)
    instrument_id: str = Field(min_length=1)
    entry_price: float = Field(gt=0)
    exit_price: float = Field(gt=0)
    side: int = Field(default=1)
    gross_return: float
    spread_cost: float = Field(ge=0)
    fee_cost: float = Field(ge=0)
    slippage_cost: float = Field(ge=0)
    liquidity_impact_cost: float = Field(ge=0)
    total_cost: float = Field(ge=0)
    net_return: float
    is_positive: bool
    label_version: str = Field(default=LABEL_VERSION, min_length=1)

    @field_validator("side")
    @classmethod
    def validate_side(cls, value: int) -> int:
        if value not in (-1, 1):
            raise ValueError("side must be -1 or 1")
        return value

    @model_validator(mode="after")
    def validate_label(self) -> Label:
        if self.label_ts < self.decision_ts + self.horizon_ms:
            raise ValueError("label_ts must be at or after decision_ts + horizon_ms")
        expected_cost = (
            self.spread_cost
            + self.fee_cost
            + self.slippage_cost
            + self.liquidity_impact_cost
        )
        if abs(self.total_cost - expected_cost) > 1e-12:
            raise ValueError("total_cost must equal component costs")
        if abs(self.net_return - (self.gross_return - self.total_cost)) > 1e-12:
            raise ValueError("net_return must equal gross_return - total_cost")
        if self.is_positive != (self.net_return > 0.0):
            raise ValueError("is_positive must equal net_return > 0")
        return self

    def to_row(self) -> dict[str, Any]:
        return self.model_dump()


MARKET_DATA_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("ts", pa.int64(), nullable=False),
        pa.field("available_at_ts", pa.int64(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("instrument_id", pa.string(), nullable=False),
        pa.field("bid_price", pa.float64(), nullable=True),
        pa.field("ask_price", pa.float64(), nullable=True),
        pa.field("mid_price", pa.float64(), nullable=True),
        pa.field("last_price", pa.float64(), nullable=True),
        pa.field("volume", pa.float64(), nullable=False),
        pa.field("trade_count", pa.int64(), nullable=False),
        pa.field("bid_size", pa.float64(), nullable=True),
        pa.field("ask_size", pa.float64(), nullable=True),
        pa.field("liquidity_depth", pa.float64(), nullable=True),
        pa.field("timeframe_ms", pa.int64(), nullable=True),
        pa.field("sequence", pa.int64(), nullable=True),
    ],
    metadata={b"bigan.contract": b"v8.phase0.MarketData"},
)

FEATURE_VECTOR_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("decision_ts", pa.int64(), nullable=False),
        pa.field("feature_cutoff_ts", pa.int64(), nullable=False),
        pa.field("lookback_start_ts", pa.int64(), nullable=False),
        pa.field("max_input_ts", pa.int64(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("instrument_id", pa.string(), nullable=False),
        pa.field("feature_version", pa.string(), nullable=False),
        *[pa.field(column, pa.float64(), nullable=True) for column in FEATURE_COLUMNS],
    ],
    metadata={b"bigan.contract": b"v8.phase0.FeatureVector"},
)

LABEL_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("decision_ts", pa.int64(), nullable=False),
        pa.field("label_ts", pa.int64(), nullable=False),
        pa.field("horizon_ms", pa.int64(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("instrument_id", pa.string(), nullable=False),
        pa.field("entry_price", pa.float64(), nullable=False),
        pa.field("exit_price", pa.float64(), nullable=False),
        pa.field("side", pa.int64(), nullable=False),
        pa.field("gross_return", pa.float64(), nullable=False),
        pa.field("spread_cost", pa.float64(), nullable=False),
        pa.field("fee_cost", pa.float64(), nullable=False),
        pa.field("slippage_cost", pa.float64(), nullable=False),
        pa.field("liquidity_impact_cost", pa.float64(), nullable=False),
        pa.field("total_cost", pa.float64(), nullable=False),
        pa.field("net_return", pa.float64(), nullable=False),
        pa.field("is_positive", pa.bool_(), nullable=False),
        pa.field("label_version", pa.string(), nullable=False),
    ],
    metadata={b"bigan.contract": b"v8.phase0.Label"},
)


class DatasetContract(BaseModel):
    """Reproducible dataset-level contract.

    This is the strict contract downstream phases should inspect before using a
    Phase 0 artifact. It binds the feature schema, label/cost schema, metadata,
    and deterministic dataset hash into one auditable object.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    dataset_version: str = Field(min_length=1)
    dataset_hash: str = Field(min_length=1)
    market_schema: tuple[str, ...]
    feature_schema: tuple[str, ...]
    label_schema: tuple[str, ...]
    cost_columns: tuple[str, ...] = COST_COLUMNS
    market_schema_hash: str | None = None
    feature_schema_hash: str | None = None
    label_schema_hash: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def derive_schema_hashes(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        row = dict(data)
        for field_name in ("market_schema", "feature_schema", "label_schema"):
            hash_field = f"{field_name}_hash"
            if row.get(hash_field) is None and row.get(field_name) is not None:
                row[hash_field] = schema_names_hash(tuple(row[field_name]))
        return row

    @model_validator(mode="after")
    def validate_contract(self) -> DatasetContract:
        _require_schema_columns(
            "market_schema",
            self.market_schema,
            tuple(MARKET_DATA_SCHEMA.names),
        )
        _require_schema_columns(
            "feature_schema",
            self.feature_schema,
            tuple(FEATURE_VECTOR_SCHEMA.names),
        )
        _require_schema_columns(
            "label_schema",
            self.label_schema,
            tuple(LABEL_SCHEMA.names),
        )
        missing_cost_columns = set(self.cost_columns) - set(self.label_schema)
        if missing_cost_columns:
            raise ValueError(
                "label_schema is missing cost columns: "
                + ", ".join(sorted(missing_cost_columns))
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @property
    def market_schema_order_matches(self) -> bool:
        return self.market_schema == tuple(MARKET_DATA_SCHEMA.names)

    @property
    def feature_schema_order_matches(self) -> bool:
        return self.feature_schema == tuple(FEATURE_VECTOR_SCHEMA.names)

    @property
    def label_schema_order_matches(self) -> bool:
        return self.label_schema == tuple(LABEL_SCHEMA.names)


def schema_names_hash(names: tuple[str, ...]) -> str:
    import hashlib
    import json

    payload = json.dumps(list(names), separators=(",", ":"), sort_keys=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _require_schema_columns(
    field_name: str,
    observed: tuple[str, ...],
    required: tuple[str, ...],
) -> None:
    missing = set(required) - set(observed)
    if missing:
        raise ValueError(
            f"{field_name} is missing required columns: "
            + ", ".join(sorted(missing))
        )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
