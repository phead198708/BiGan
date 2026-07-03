"""Contracts for Polymarket BTC Up/Down policy training artifacts."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from bigan.v8.polymarket.contracts import canonical_json_sha256, looks_like_sha256

POLYMARKET_POLICY_SCHEMA_VERSION = "bigan-v8-polymarket-policy-v1"
POLYMARKET_POLICY_TRAINING_PHASE = "polymarket_policy_training"
POLYMARKET_POLICY_SIGNAL_SOURCE_TRAINED_MODEL = "trained_model"
DEFAULT_POLICY_CREATED_AT = "1970-01-01T00:00:00Z"
DEFAULT_ACTION_VALUE_MODEL_VERSION = "polymarket_action_value_policy_v1"

PolicyAction = Literal["BUY_UP", "BUY_DOWN", "SELL_UP", "SELL_DOWN", "HOLD", "NO_TRADE"]
PolicyOutcome = Literal["UP", "DOWN", "NO_TRADE"]
PolicyLabelAction = Literal[
    "NO_TRADE",
    "BUY_UP_HOLD_TO_SETTLEMENT",
    "BUY_DOWN_HOLD_TO_SETTLEMENT",
    "BUY_UP_SELL_BEFORE_CLOSE",
    "BUY_DOWN_SELL_BEFORE_CLOSE",
]

ACTION_VALUE_LABEL_ACTIONS: tuple[PolicyLabelAction, ...] = (
    "NO_TRADE",
    "BUY_UP_HOLD_TO_SETTLEMENT",
    "BUY_DOWN_HOLD_TO_SETTLEMENT",
    "BUY_UP_SELL_BEFORE_CLOSE",
    "BUY_DOWN_SELL_BEFORE_CLOSE",
)
PRIMARY_POLICY_TARGET_ACTION_VALUE = "action_expected_net_return"
AUXILIARY_OUTCOME_TARGET = "resolved_up_probability"


@dataclass(frozen=True, slots=True)
class PolymarketPolicyTrainingConfig:
    """Configuration for deterministic offline Polymarket policy training."""

    corpus_dir: Path | str
    output_dir: Path | str
    run_id: str = "polymarket_policy_fixture_run"
    model_version: str = DEFAULT_ACTION_VALUE_MODEL_VERSION
    created_at: str = DEFAULT_POLICY_CREATED_AT
    train_fraction: float = 0.60
    validation_fraction: float = 0.25
    ev_threshold: float = 0.015
    min_confidence: float = 0.05
    max_paper_notional: float = 0.20
    fee_rate: float = 0.0002
    slippage_rate: float = 0.0005
    liquidity_impact_rate: float = 0.0001
    sell_before_close_exit_buffer_seconds: int = 30
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
        if self.sell_before_close_exit_buffer_seconds <= 0:
            raise ValueError("sell_before_close_exit_buffer_seconds must be positive")
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
    action_return_targets: dict[str, float] = field(default_factory=dict)
    realized_trade_return_targets: dict[str, float] = field(default_factory=dict)
    settlement_return_targets: dict[str, float] = field(default_factory=dict)
    action_is_positive_targets: dict[str, bool] = field(default_factory=dict)
    sell_before_close_execution_class_targets: dict[str, str] = field(default_factory=dict)
    sell_before_close_theoretical_return_targets: dict[str, float] = field(default_factory=dict)
    sell_before_close_executable_return_targets: dict[str, float] = field(default_factory=dict)
    sell_before_close_execution_gap_targets: dict[str, float] = field(default_factory=dict)
    sell_before_close_queue_fill_probability_targets: dict[str, float] = field(
        default_factory=dict
    )
    sell_before_close_exit_bid_targets: dict[str, float] = field(default_factory=dict)
    sell_before_close_executable_liquidity_notional_targets: dict[str, float] = field(
        default_factory=dict
    )
    sell_before_close_exit_path_targets: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )
    sell_before_close_label_uses_executable_exit_path_targets: dict[str, bool] = field(
        default_factory=dict
    )
    best_policy_action: str = "NO_TRADE"
    best_action_expected_return: float = 0.0
    second_best_action_expected_return: float = 0.0
    best_action_margin: float = 0.0
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
        _validate_action_return_targets(self.action_return_targets, allow_empty=True)
        _validate_numeric_target_mapping(
            self.realized_trade_return_targets,
            allow_empty=True,
            field_name="realized_trade_return_targets",
        )
        _validate_numeric_target_mapping(
            self.settlement_return_targets,
            allow_empty=True,
            field_name="settlement_return_targets",
        )
        if self.action_is_positive_targets:
            unsupported = set(self.action_is_positive_targets) - set(ACTION_VALUE_LABEL_ACTIONS)
            if unsupported:
                raise ValueError(
                    "action_is_positive_targets contains unsupported actions: "
                    + ", ".join(sorted(unsupported))
                )
        for mapping, field_name in (
            (
                self.sell_before_close_theoretical_return_targets,
                "sell_before_close_theoretical_return_targets",
            ),
            (
                self.sell_before_close_executable_return_targets,
                "sell_before_close_executable_return_targets",
            ),
            (
                self.sell_before_close_execution_gap_targets,
                "sell_before_close_execution_gap_targets",
            ),
            (
                self.sell_before_close_queue_fill_probability_targets,
                "sell_before_close_queue_fill_probability_targets",
            ),
            (
                self.sell_before_close_exit_bid_targets,
                "sell_before_close_exit_bid_targets",
            ),
            (
                self.sell_before_close_executable_liquidity_notional_targets,
                "sell_before_close_executable_liquidity_notional_targets",
            ),
        ):
            _validate_numeric_target_mapping(
                mapping,
                allow_empty=True,
                field_name=field_name,
            )
        for action in self.sell_before_close_execution_class_targets:
            if action not in ACTION_VALUE_LABEL_ACTIONS:
                raise ValueError("unsupported sell-before-close execution class action")
        for action, exit_path in self.sell_before_close_exit_path_targets.items():
            if action not in ACTION_VALUE_LABEL_ACTIONS:
                raise ValueError("unsupported sell-before-close exit path action")
            if not isinstance(exit_path, dict):
                raise ValueError("sell-before-close exit path targets must be dicts")
        for action in self.sell_before_close_label_uses_executable_exit_path_targets:
            if action not in ACTION_VALUE_LABEL_ACTIONS:
                raise ValueError("unsupported sell-before-close executable path action")
        if self.best_policy_action not in ACTION_VALUE_LABEL_ACTIONS:
            raise ValueError("best_policy_action must be a supported label action")
        for field_name in (
            "best_action_expected_return",
            "second_best_action_expected_return",
            "best_action_margin",
        ):
            if not math.isfinite(float(getattr(self, field_name))):
                raise ValueError(f"{field_name} must be finite")
        if self.best_action_margin < -1e-12:
            raise ValueError("best_action_margin must be non-negative")
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
    split_metadata: dict[str, Any]
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
        _validate_split_metadata(self.split_metadata)
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
    p_up_auxiliary: float | None = None
    expected_return_by_action: dict[str, float] = field(default_factory=dict)
    expected_return_no_trade: float | None = None
    expected_return_buy_up_hold_to_settlement: float | None = None
    expected_return_buy_down_hold_to_settlement: float | None = None
    expected_return_buy_up_sell_before_close: float | None = None
    expected_return_buy_down_sell_before_close: float | None = None
    best_policy_action: str | None = None
    best_action_expected_return: float | None = None
    second_best_action_expected_return: float | None = None
    best_action_margin: float | None = None
    calibrated_expected_pnl_per_notional_by_action: dict[str, float] = field(
        default_factory=dict
    )
    calibrated_best_policy_action: str | None = None
    calibrated_expected_pnl_per_notional: float | None = None
    calibrated_second_best_expected_pnl_per_notional: float | None = None
    calibrated_action_margin: float | None = None
    action_value_calibration_applied: bool = False
    action_value_calibration_id: str | None = None
    calibration_support_count: int | None = None
    calibration_bucket_count: int | None = None
    policy_confidence: float | None = None
    action_value_head_enabled: bool = False
    outcome_probability_head_enabled: bool = True
    action_value_model_family: str = "resolved_up_probability_only"
    feature_conditioned_action_value_model_enabled: bool = False
    paper_only: bool = True
    capital_at_risk: bool = False
    polymarket_write_enabled: bool = False
    wallet_signing_enabled: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.estimated_up_probability <= 1.0:
            raise ValueError("estimated_up_probability must be in [0, 1]")
        if self.p_up_auxiliary is not None and not 0.0 <= self.p_up_auxiliary <= 1.0:
            raise ValueError("p_up_auxiliary must be in [0, 1]")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.policy_confidence is not None and not 0.0 <= self.policy_confidence <= 1.0:
            raise ValueError("policy_confidence must be in [0, 1]")
        if not self.action_value_model_family.strip():
            raise ValueError("action_value_model_family is required")
        if not math.isfinite(self.score):
            raise ValueError("score must be finite")
        _validate_action_return_targets(
            self.expected_return_by_action,
            allow_empty=not self.action_value_head_enabled,
        )
        if self.action_value_head_enabled:
            if self.best_policy_action not in ACTION_VALUE_LABEL_ACTIONS:
                raise ValueError("best_policy_action must be present for action-value output")
            for field_name in (
                "best_action_expected_return",
                "second_best_action_expected_return",
                "best_action_margin",
                "expected_return_no_trade",
                "expected_return_buy_up_hold_to_settlement",
                "expected_return_buy_down_hold_to_settlement",
                "expected_return_buy_up_sell_before_close",
                "expected_return_buy_down_sell_before_close",
            ):
                value = getattr(self, field_name)
                if value is None or not math.isfinite(float(value)):
                    raise ValueError(f"{field_name} must be finite for action-value output")
        if self.action_value_calibration_applied:
            _validate_action_return_targets(
                self.calibrated_expected_pnl_per_notional_by_action,
                allow_empty=False,
            )
            if self.calibrated_best_policy_action not in ACTION_VALUE_LABEL_ACTIONS:
                raise ValueError(
                    "calibrated_best_policy_action must be present for calibrated output"
                )
            for field_name in (
                "calibrated_expected_pnl_per_notional",
                "calibrated_second_best_expected_pnl_per_notional",
                "calibrated_action_margin",
                "calibration_support_count",
                "calibration_bucket_count",
            ):
                value = getattr(self, field_name)
                if value is None or not math.isfinite(float(value)):
                    raise ValueError(f"{field_name} must be finite for calibrated output")
            if self.calibrated_action_margin is not None and self.calibrated_action_margin < -1e-12:
                raise ValueError("calibrated_action_margin must be non-negative")
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
    primary_policy_target: str = PRIMARY_POLICY_TARGET_ACTION_VALUE
    outcome_probability_head_enabled: bool = True
    action_value_head_enabled: bool = False
    compatibility_probability_fallback_enabled: bool = True
    action_value_model_family: str = "market_family_mean_baseline"
    fallback_action_value_model_family: str = "market_family_mean_baseline"
    feature_conditioned_action_value_model_enabled: bool = False
    action_value_feature_columns: tuple[str, ...] = ()
    action_return_feature_means: dict[str, float] = field(default_factory=dict)
    action_return_feature_coefficients: dict[str, dict[str, float]] = field(default_factory=dict)
    global_action_returns: dict[str, float] = field(default_factory=dict)
    market_family_action_returns: dict[str, dict[str, float]] = field(default_factory=dict)
    family_action_feature_offsets: dict[str, dict[str, float]] = field(default_factory=dict)
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
        if self.primary_policy_target not in (
            PRIMARY_POLICY_TARGET_ACTION_VALUE,
            AUXILIARY_OUTCOME_TARGET,
            "resolved_up_probability_only",
        ):
            raise ValueError("unsupported primary_policy_target")
        _require_non_empty(
            action_value_model_family=self.action_value_model_family,
            fallback_action_value_model_family=self.fallback_action_value_model_family,
        )
        if self.feature_conditioned_action_value_model_enabled and not self.action_value_feature_columns:
            raise ValueError("action_value_feature_columns are required for feature-conditioned model")
        for feature_name in self.action_value_feature_columns:
            if not feature_name.strip():
                raise ValueError("action_value_feature_columns must be non-empty")
        for feature_name, value in self.action_return_feature_means.items():
            if not feature_name.strip() or not math.isfinite(float(value)):
                raise ValueError("action_return_feature_means must be finite")
        for action, coefficients in self.action_return_feature_coefficients.items():
            if action not in ACTION_VALUE_LABEL_ACTIONS:
                raise ValueError("action_return_feature_coefficients contains unsupported action")
            for feature_name, value in coefficients.items():
                if not feature_name.strip() or not math.isfinite(float(value)):
                    raise ValueError("action_return_feature_coefficients must be finite")
        _validate_action_return_targets(
            self.global_action_returns,
            allow_empty=not self.action_value_head_enabled,
        )
        for family, returns in self.market_family_action_returns.items():
            if not family.strip():
                raise ValueError("market_family_action_returns keys must be non-empty")
            _validate_action_return_targets(returns, allow_empty=False)
        for family, offsets in self.family_action_feature_offsets.items():
            if not family.strip():
                raise ValueError("family_action_feature_offsets keys must be non-empty")
            _validate_numeric_target_mapping(
                offsets,
                allow_empty=True,
                field_name="family_action_feature_offsets",
            )
        _validate_safety_boundary(self)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PolymarketPolicyTrainingResult:
    run_dir: Path
    dataset: PolymarketPolicyDataset
    model: PolymarketPolicyModel
    predictions: tuple[PolymarketPolicyPrediction, ...]
    train_predictions: tuple[PolymarketPolicyPrediction, ...]
    validation_predictions: tuple[PolymarketPolicyPrediction, ...]
    shadow_predictions: tuple[PolymarketPolicyPrediction, ...]
    artifact_paths: dict[str, Path]
    artifact_hashes: dict[str, str]
    model_manifest: dict[str, Any]
    calibration_report: dict[str, Any]
    validation_report: dict[str, Any]
    ev_threshold_report: dict[str, Any]
    replay_report: dict[str, Any]
    action_value_signal_sanity_report: dict[str, Any]
    action_family_eligibility_report: dict[str, Any]
    hold_to_settlement_longshot_guard_report: dict[str, Any]
    action_family_replay_variants_report: dict[str, Any]
    action_family_counterfactual_replay_report: dict[str, Any]
    model_ranking_error_report: dict[str, Any]
    model_ranking_candidate_comparison_report: dict[str, Any]
    action_representation_diagnostic_report: dict[str, Any]
    ranking_overlay_zero_entry_diagnostic_report: dict[str, Any]
    source_model_eligibility_report: dict[str, Any]
    sell_before_close_p_up_disagreement_diagnostic_report: dict[str, Any]
    sell_before_close_exit_reliability_report: dict[str, Any]
    sell_before_close_promotion_support_gate_report: dict[str, Any]
    sell_before_close_support_aware_threshold_selection_report: dict[str, Any]
    sell_before_close_support_aware_threshold_failure_attribution_report: dict[
        str,
        Any,
    ]
    sell_before_close_validation_failure_drilldown_report: dict[str, Any]
    sell_before_close_guard_threshold_sweep_report: dict[str, Any]
    m_frozen_selector_walk_forward_report: dict[str, Any]


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


def _validate_split_metadata(split_metadata: dict[str, Any]) -> None:
    required = (
        "train_min_ts",
        "train_max_ts",
        "validation_min_ts",
        "validation_max_ts",
        "shadow_min_ts",
        "shadow_max_ts",
    )
    missing = [field_name for field_name in required if field_name not in split_metadata]
    if missing:
        raise ValueError(f"split_metadata missing fields: {', '.join(missing)}")
    train_max = int(split_metadata["train_max_ts"])
    validation_min = int(split_metadata["validation_min_ts"])
    validation_max = int(split_metadata["validation_max_ts"])
    shadow_min = int(split_metadata["shadow_min_ts"])
    if train_max >= validation_min:
        raise ValueError("train split must strictly precede validation split")
    if validation_max >= shadow_min:
        raise ValueError("validation split must strictly precede shadow split")


def _validate_action_return_targets(
    values: dict[str, float],
    *,
    allow_empty: bool,
) -> None:
    _validate_numeric_target_mapping(
        values,
        allow_empty=allow_empty,
        field_name="action_return_targets",
    )
    if values:
        missing = set(ACTION_VALUE_LABEL_ACTIONS) - set(values)
        if missing:
            raise ValueError(
                "action return targets missing actions: " + ", ".join(sorted(missing))
            )


def _validate_numeric_target_mapping(
    values: dict[str, float],
    *,
    allow_empty: bool,
    field_name: str,
) -> None:
    if not values:
        if allow_empty:
            return
        raise ValueError(f"{field_name} must not be empty")
    unsupported = set(values) - set(ACTION_VALUE_LABEL_ACTIONS)
    if unsupported:
        raise ValueError(
            f"{field_name} contains unsupported actions: " + ", ".join(sorted(unsupported))
        )
    for action, value in values.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"{field_name}.{action} must be finite")
