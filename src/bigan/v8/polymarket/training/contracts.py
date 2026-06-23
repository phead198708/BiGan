"""Contracts for Polymarket BTC Up/Down policy training artifacts."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from bigan.v8.polymarket.contracts import canonical_json_sha256, looks_like_sha256

POLYMARKET_POLICY_SCHEMA_VERSION = "bigan-v8-polymarket-policy-v1"
POLYMARKET_POLICY_TRAINING_PHASE = "polymarket_policy_training"
POLYMARKET_POLICY_SIGNAL_SOURCE_TRAINED_MODEL = "trained_model"
DEFAULT_POLICY_CREATED_AT = "1970-01-01T00:00:00Z"

PolicyAction = Literal["BUY_UP", "BUY_DOWN", "SELL_UP", "SELL_DOWN", "HOLD", "NO_TRADE"]
PolicyOutcome = Literal["UP", "DOWN", "NO_TRADE"]


@dataclass(frozen=True, slots=True)
class PolymarketPolicyTrainingConfig:
    """Configuration for deterministic offline Polymarket policy training."""

    corpus_dir: Path | str
    output_dir: Path | str
    run_id: str = "polymarket_policy_fixture_run"
    model_version: str = "polymarket_policy_probability_v1"
    created_at: str = DEFAULT_POLICY_CREATED_AT
    train_fraction: float = 0.60
    validation_fraction: float = 0.25
    ev_threshold: float = 0.015
    min_confidence: float = 0.05
    max_paper_notional: float = 0.20
    fee_rate: float = 0.0002
    slippage_rate: float = 0.0005
    liquidity_impact_rate: float = 0.0001
    overwrite_existing: bool = False
    paper_only: bool = True
    capital_at_risk: bool = False
    broker_exchange_write_enabled: bool = False
    live_exchange_write_enabled: bool = False
    polymarket_write_enabled: bool = False
    wallet_signing_enabled: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.corpus_dir, Path):
            object.__setattr__(self, "corpus_dir", Path(self.corpus_dir))
        if not isinstance(self.output_dir, Path):
            object.__setattr__(self, "output_dir", Path(self.output_dir))
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if not self.model_version.strip():
            raise ValueError("model_version is required")
        if not self.created_at:
            raise ValueError("created_at is required")
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must be in (0, 1)")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")
        if self.train_fraction + self.validation_fraction >= 1.0:
            raise ValueError("train_fraction + validation_fraction must leave shadow rows")
        if self.ev_threshold < 0.0:
            raise ValueError("ev_threshold must be non-negative")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        for field_name in (
            "max_paper_notional",
            "fee_rate",
            "slippage_rate",
            "liquidity_impact_rate",
        ):
            value = float(getattr(self, field_name))
            if value < 0.0 or not math.isfinite(value):
                raise ValueError(f"{field_name} must be non-negative and finite")
        _validate_safety_boundary(self)

    @property
    def run_dir(self) -> Path:
        return self.output_dir.expanduser().resolve() / self.run_id

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["corpus_dir"] = str(self.corpus_dir)
        payload["output_dir"] = str(self.output_dir)
        return payload

    def to_manifest_dict(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload.pop("corpus_dir", None)
        payload.pop("output_dir", None)
        return payload


@dataclass(frozen=True, slots=True)
class PolymarketPolicyExample:
    market_id: str
    condition_id: str
    slug: str
    market_family: str
    horizon_ms: int
    decision_ts: int
    feature_cutoff_ts: int
    max_input_ts: int
    features: dict[str, float]
    target_up_probability: float
    resolved_outcome: str
    resolution_status: str
    paper_only: bool = True
    capital_at_risk: bool = False
    polymarket_write_enabled: bool = False
    wallet_signing_enabled: bool = False

    def __post_init__(self) -> None:
        _require_non_empty(
            market_id=self.market_id,
            condition_id=self.condition_id,
            slug=self.slug,
            market_family=self.market_family,
            resolved_outcome=self.resolved_outcome,
            resolution_status=self.resolution_status,
        )
        if self.horizon_ms <= 0:
            raise ValueError("horizon_ms must be positive")
        if self.feature_cutoff_ts > self.decision_ts or self.max_input_ts > self.decision_ts:
            raise ValueError("policy examples must preserve feature causality")
        if not 0.0 <= self.target_up_probability <= 1.0:
            raise ValueError("target_up_probability must be in [0, 1]")
        if not self.features:
            raise ValueError("features must not be empty")
        for name, value in self.features.items():
            if not name.strip():
                raise ValueError("feature names must be non-empty")
            if not math.isfinite(float(value)):
                raise ValueError("feature values must be finite")
        _validate_safety_boundary(self)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PolymarketPolicyDataset:
    examples: tuple[PolymarketPolicyExample, ...]
    feature_columns: tuple[str, ...]
    feature_schema_hash: str
    label_schema_hash: str
    training_corpus_hash: str
    dataset_hash: str
    corpus_manifest: dict[str, Any]
    market_metadata: dict[str, dict[str, Any]]
    resolution_events: dict[str, dict[str, Any]]
    train_examples: tuple[PolymarketPolicyExample, ...]
    validation_examples: tuple[PolymarketPolicyExample, ...]
    shadow_examples: tuple[PolymarketPolicyExample, ...]
    paper_only: bool = True
    capital_at_risk: bool = False
    polymarket_write_enabled: bool = False
    wallet_signing_enabled: bool = False

    def __post_init__(self) -> None:
        if not self.examples:
            raise ValueError("policy dataset examples must not be empty")
        if not self.feature_columns:
            raise ValueError("feature_columns must not be empty")
        for field_name in ("feature_schema_hash", "label_schema_hash", "training_corpus_hash", "dataset_hash"):
            if not looks_like_sha256(getattr(self, field_name)):
                raise ValueError(f"{field_name} must be SHA-256")
        if not self.train_examples or not self.validation_examples or not self.shadow_examples:
            raise ValueError("train, validation, and shadow examples must be non-empty")
        _validate_safety_boundary(self)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["examples"] = [example.to_dict() for example in self.examples]
        payload["train_examples"] = [example.to_dict() for example in self.train_examples]
        payload["validation_examples"] = [
            example.to_dict() for example in self.validation_examples
        ]
        payload["shadow_examples"] = [example.to_dict() for example in self.shadow_examples]
        return payload


@dataclass(frozen=True, slots=True)
class PolymarketPolicyPrediction:
    market_id: str
    condition_id: str
    slug: str
    market_family: str
    horizon_ms: int
    decision_ts: int
    estimated_up_probability: float
    confidence: float
    score: float
    calibration_bucket: str
    model_version: str
    feature_schema_hash: str
    training_corpus_hash: str
    features: dict[str, float]
    target_up_probability: float | None = None
    paper_only: bool = True
    capital_at_risk: bool = False
    polymarket_write_enabled: bool = False
    wallet_signing_enabled: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.estimated_up_probability <= 1.0:
            raise ValueError("estimated_up_probability must be in [0, 1]")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if not math.isfinite(self.score):
            raise ValueError("score must be finite")
        if not looks_like_sha256(self.feature_schema_hash):
            raise ValueError("feature_schema_hash must be SHA-256")
        if not looks_like_sha256(self.training_corpus_hash):
            raise ValueError("training_corpus_hash must be SHA-256")
        _validate_safety_boundary(self)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PolymarketPolicyModel:
    model_version: str
    feature_columns: tuple[str, ...]
    global_probability: float
    market_family_probabilities: dict[str, float]
    family_feature_offsets: dict[str, float]
    feature_schema_hash: str
    label_schema_hash: str
    training_corpus_hash: str
    dataset_hash: str
    train_row_count: int
    paper_only: bool = True
    capital_at_risk: bool = False
    polymarket_write_enabled: bool = False
    wallet_signing_enabled: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.global_probability <= 1.0:
            raise ValueError("global_probability must be in [0, 1]")
        for field_name in ("feature_schema_hash", "label_schema_hash", "training_corpus_hash", "dataset_hash"):
            if not looks_like_sha256(getattr(self, field_name)):
                raise ValueError(f"{field_name} must be SHA-256")
        if self.train_row_count <= 0:
            raise ValueError("train_row_count must be positive")
        _validate_safety_boundary(self)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PolymarketPolicyTrainingResult:
    run_dir: Path
    dataset: PolymarketPolicyDataset
    model: PolymarketPolicyModel
    predictions: tuple[PolymarketPolicyPrediction, ...]
    artifact_paths: dict[str, Path]
    artifact_hashes: dict[str, str]
    model_manifest: dict[str, Any]
    calibration_report: dict[str, Any]
    validation_report: dict[str, Any]
    ev_threshold_report: dict[str, Any]
    replay_report: dict[str, Any]


def safety_fields() -> dict[str, bool]:
    return {
        "paper_only": True,
        "capital_at_risk": False,
        "broker_exchange_write_enabled": False,
        "live_exchange_write_enabled": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }


def compact_safety_fields() -> dict[str, bool]:
    return {
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }


def stable_hash(payload: Any) -> str:
    return canonical_json_sha256(payload)


def _require_non_empty(**values: str) -> None:
    for field_name, value in values.items():
        if not str(value).strip():
            raise ValueError(f"{field_name} is required")


def _validate_safety_boundary(payload: Any) -> None:
    for field_name, expected in compact_safety_fields().items():
        if getattr(payload, field_name) is not expected:
            raise ValueError(f"{field_name} must be {str(expected).lower()}")
