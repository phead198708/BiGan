"""Phase 5 safety-layer contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PHASE5_SAFETY_LAYER_PHASE = "phase5_safety_layer_shadow_rollback"
DEFAULT_PHASE5_CREATED_AT = "1970-01-01T00:00:00Z"


class Phase5SafetyError(RuntimeError):
    """Raised when Phase 5 receives unsafe streams or cannot fail closed."""


@dataclass(frozen=True, slots=True)
class SafetyLayerConfig:
    """Thresholds and output settings for Phase 5 safety monitoring."""

    detection_window_size: int = 5
    min_shadow_live_correlation: float = 0.80
    max_mean_pnl_drift: float = 0.006
    max_cost_drift_ratio: float = 0.50
    max_regime_mismatch_rate: float = 0.20
    max_live_drawdown: float = 0.05
    cost_drift_floor: float = 1e-6
    flatten_positions_on_kill: bool = True
    freeze_model_updates_on_kill: bool = True
    output_dir: Path | str | None = None
    created_at: str = DEFAULT_PHASE5_CREATED_AT

    def __post_init__(self) -> None:
        if self.output_dir is not None and not isinstance(self.output_dir, Path):
            object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.detection_window_size <= 1:
            raise ValueError("detection_window_size must be greater than 1")
        if not -1.0 <= self.min_shadow_live_correlation <= 1.0:
            raise ValueError("min_shadow_live_correlation must be in [-1, 1]")
        if self.max_mean_pnl_drift < 0.0:
            raise ValueError("max_mean_pnl_drift must be non-negative")
        if self.max_cost_drift_ratio < 0.0:
            raise ValueError("max_cost_drift_ratio must be non-negative")
        if not 0.0 <= self.max_regime_mismatch_rate <= 1.0:
            raise ValueError("max_regime_mismatch_rate must be in [0, 1]")
        if self.max_live_drawdown <= 0.0:
            raise ValueError("max_live_drawdown must be positive")
        if self.cost_drift_floor <= 0.0:
            raise ValueError("cost_drift_floor must be positive")
        if not self.created_at:
            raise ValueError("created_at is required")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_dir"] = None if self.output_dir is None else str(self.output_dir)
        return payload


@dataclass(frozen=True, slots=True)
class LiveExecutionObservation:
    """One live execution observation paired with a shadow decision."""

    decision_ts: int
    source: str
    instrument_id: str
    live_filled_action: float
    live_net_return: float
    live_total_execution_cost: float
    live_regime: str
    capital_at_risk: bool = True

    def __post_init__(self) -> None:
        if self.decision_ts < 0:
            raise ValueError("decision_ts must be non-negative")
        if not self.source:
            raise ValueError("source is required")
        if not self.instrument_id:
            raise ValueError("instrument_id is required")
        if not 0.0 <= self.live_filled_action <= 1.0:
            raise ValueError("live_filled_action must be in [0, 1]")
        for field_name in ("live_net_return", "live_total_execution_cost"):
            if not math.isfinite(float(getattr(self, field_name))):
                raise ValueError(f"{field_name} must be finite")
        if self.live_total_execution_cost < 0.0:
            raise ValueError("live_total_execution_cost must be non-negative")
        if not self.live_regime:
            raise ValueError("live_regime is required")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ShadowLiveRecord:
    """Auditable pair of shadow simulation output and live execution result."""

    decision_ts: int
    source: str
    instrument_id: str
    shadow_net_return: float
    live_net_return: float
    shadow_total_execution_cost: float
    live_total_execution_cost: float
    shadow_regime: str
    live_regime: str
    shadow_filled_action: float
    live_filled_action: float
    shadow_capital_at_risk: bool
    live_capital_at_risk: bool

    def __post_init__(self) -> None:
        for field_name in (
            "shadow_net_return",
            "live_net_return",
            "shadow_total_execution_cost",
            "live_total_execution_cost",
            "shadow_filled_action",
            "live_filled_action",
        ):
            if not math.isfinite(float(getattr(self, field_name))):
                raise ValueError(f"{field_name} must be finite")
        if self.shadow_capital_at_risk:
            raise ValueError("shadow_capital_at_risk must be false")
        if self.shadow_total_execution_cost < 0.0:
            raise ValueError("shadow_total_execution_cost must be non-negative")
        if self.live_total_execution_cost < 0.0:
            raise ValueError("live_total_execution_cost must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class StableModelSnapshot:
    """Rollback target known to be safe before Phase 5 monitoring starts."""

    model_id: str
    model_sha256: str
    policy_dataset_hash: str
    split_hash: str
    safe_parameter_sha256: str
    safe_parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id is required")
        for field_name in (
            "model_sha256",
            "policy_dataset_hash",
            "split_hash",
            "safe_parameter_sha256",
        ):
            value = getattr(self, field_name)
            if not _looks_like_sha256(value):
                raise ValueError(f"{field_name} must be a SHA-256 hex digest")
        if not self.safe_parameters:
            raise ValueError("safe_parameters must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_sha256": self.model_sha256,
            "policy_dataset_hash": self.policy_dataset_hash,
            "split_hash": self.split_hash,
            "safe_parameter_sha256": self.safe_parameter_sha256,
            "safe_parameters": dict(self.safe_parameters),
        }


@dataclass(frozen=True, slots=True)
class SafetyAction:
    """Fail-closed action emitted by the safety layer."""

    kill_switch_triggered: bool
    stop_trading: bool
    flatten_positions: bool
    freeze_model_updates: bool
    rollback_model_id: str | None
    rollback_model_sha256: str | None
    restored_safe_parameters: dict[str, Any]
    reason_codes: tuple[str, ...]
    triggered_at_ts: int | None = None

    def __post_init__(self) -> None:
        if self.kill_switch_triggered:
            if not self.stop_trading:
                raise ValueError("kill switch must stop trading")
            if not self.rollback_model_id:
                raise ValueError("kill switch requires rollback_model_id")
            if not self.rollback_model_sha256:
                raise ValueError("kill switch requires rollback_model_sha256")
            if not self.restored_safe_parameters:
                raise ValueError("kill switch requires restored_safe_parameters")
            if not self.reason_codes:
                raise ValueError("kill switch requires reason_codes")
        if self.rollback_model_sha256 is not None and not _looks_like_sha256(
            self.rollback_model_sha256
        ):
            raise ValueError("rollback_model_sha256 must be a SHA-256 hex digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kill_switch_triggered": self.kill_switch_triggered,
            "stop_trading": self.stop_trading,
            "flatten_positions": self.flatten_positions,
            "freeze_model_updates": self.freeze_model_updates,
            "rollback_model_id": self.rollback_model_id,
            "rollback_model_sha256": self.rollback_model_sha256,
            "restored_safe_parameters": dict(self.restored_safe_parameters),
            "reason_codes": list(self.reason_codes),
            "triggered_at_ts": self.triggered_at_ts,
        }


@dataclass(frozen=True, slots=True)
class Phase5SafetyLayerReport:
    """Auditable Phase 5 shadow-mode, kill-switch, and rollback report."""

    phase: str
    row_count: int
    shadow_mode_metrics: dict[str, Any]
    drift_metrics: dict[str, Any]
    live_risk_metrics: dict[str, Any]
    rolling_diagnostics: list[dict[str, Any]]
    safety_action: dict[str, Any]
    rollback_snapshot: dict[str, Any]
    acceptance_criteria: dict[str, bool]
    config: dict[str, Any]
    created_at: str

    @property
    def passed(self) -> bool:
        return all(self.acceptance_criteria.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "passed": self.passed,
            "row_count": self.row_count,
            "shadow_mode_metrics": self.shadow_mode_metrics,
            "drift_metrics": self.drift_metrics,
            "live_risk_metrics": self.live_risk_metrics,
            "rolling_diagnostics": self.rolling_diagnostics,
            "safety_action": self.safety_action,
            "rollback_snapshot": self.rollback_snapshot,
            "acceptance_criteria": self.acceptance_criteria,
            "config": self.config,
            "created_at": self.created_at,
        }


def _looks_like_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value.lower())
