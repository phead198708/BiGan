"""Strategy Discovery contracts for v8 paper-only validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

from bigan.v8.paper.contracts import canonical_payload_sha256, looks_like_sha256

STRATEGY_DISCOVERY_SCHEMA_VERSION = "bigan-v8-strategy-discovery-v1"
STRATEGY_CANDIDATE_MANIFEST_PHASE = "strategy_candidate_manifest"
DEFAULT_STRATEGY_DISCOVERY_CREATED_AT = "2026-06-22T07:00:00Z"

CandidateStatus = Literal[
    "candidate_registered",
    "candidate_invalid",
    "paper_replay_passed",
    "paper_replay_failed",
    "phase5_blocked",
    "phase6_blocked_fail_closed",
    "observability_warning",
    "observability_critical",
    "ready_for_manual_review",
]


class StrategyDiscoveryError(RuntimeError):
    """Raised when strategy discovery inputs cannot be safely validated."""


@dataclass(frozen=True, slots=True)
class StrategyCandidate:
    """Normalized strategy discovery output before paper replay."""

    candidate_id: str
    candidate_family: str
    strategy_name: str
    created_at: str
    source: str
    source_commit_sha: str
    source_artifact_sha256: str
    feature_contract_sha256: str
    dataset_contract_sha256: str
    policy_config: Mapping[str, Any]
    execution_config: Mapping[str, Any]
    risk_config: Mapping[str, Any]
    expected_instruments: Sequence[str]
    expected_regime_keys: Sequence[str]
    paper_only: bool = True
    capital_at_risk: bool = False
    broker_exchange_write_enabled: bool = False
    live_exchange_write_enabled: bool = False

    def __post_init__(self) -> None:
        _validate_identifier(self.candidate_id, "candidate_id")
        for field_name in (
            "candidate_family",
            "strategy_name",
            "created_at",
            "source",
            "source_commit_sha",
        ):
            if not str(getattr(self, field_name)).strip():
                raise StrategyDiscoveryError(f"{field_name} is required")
        for field_name in (
            "source_artifact_sha256",
            "feature_contract_sha256",
            "dataset_contract_sha256",
        ):
            if not looks_like_sha256(str(getattr(self, field_name))):
                raise StrategyDiscoveryError(
                    f"{field_name} must be a SHA-256 hex digest"
                )
        for field_name in ("policy_config", "execution_config", "risk_config"):
            if not isinstance(getattr(self, field_name), Mapping):
                raise StrategyDiscoveryError(f"{field_name} must be a mapping")
        for field_name in ("expected_instruments", "expected_regime_keys"):
            values = tuple(str(value) for value in getattr(self, field_name))
            if not values or any(not value.strip() for value in values):
                raise StrategyDiscoveryError(f"{field_name} must be non-empty")
            object.__setattr__(self, field_name, values)
        if self.paper_only is not True:
            raise StrategyDiscoveryError("candidate must preserve paper_only=true")
        if self.capital_at_risk is not False:
            raise StrategyDiscoveryError("candidate cannot put capital at risk")
        if self.broker_exchange_write_enabled:
            raise StrategyDiscoveryError("broker/exchange write flag is forbidden")
        if self.live_exchange_write_enabled:
            raise StrategyDiscoveryError("live exchange write flag is forbidden")
        object.__setattr__(self, "policy_config", dict(self.policy_config))
        object.__setattr__(self, "execution_config", dict(self.execution_config))
        object.__setattr__(self, "risk_config", dict(self.risk_config))

    @property
    def candidate_sha256(self) -> str:
        return canonical_payload_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["policy_config"] = dict(sorted(self.policy_config.items()))
        payload["execution_config"] = dict(sorted(self.execution_config.items()))
        payload["risk_config"] = dict(sorted(self.risk_config.items()))
        payload["expected_instruments"] = list(self.expected_instruments)
        payload["expected_regime_keys"] = list(self.expected_regime_keys)
        return payload


@dataclass(frozen=True, slots=True)
class StrategyCandidateManifest:
    """Deterministic manifest for one strategy candidate replay."""

    schema_version: str
    phase: str
    candidate_id: str
    candidate_sha256: str
    candidate_payload: dict[str, Any]
    input_artifact_hashes: dict[str, str]
    expected_phase0_contract: str
    expected_phase1_policy_contract: str
    expected_phase2_execution_contract: str
    paper_pipeline_config: dict[str, Any]
    created_at: str

    @property
    def manifest_sha256(self) -> str:
        return canonical_payload_sha256(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = asdict(self)
        if include_hash:
            payload["manifest_sha256"] = self.manifest_sha256
        return payload


def strategy_candidate_from_mapping(payload: Mapping[str, Any]) -> StrategyCandidate:
    """Normalize a raw mapping into a validated strategy candidate."""

    return StrategyCandidate(
        candidate_id=str(payload.get("candidate_id", "")),
        candidate_family=str(payload.get("candidate_family", "")),
        strategy_name=str(payload.get("strategy_name", "")),
        created_at=str(payload.get("created_at", "")),
        source=str(payload.get("source", "")),
        source_commit_sha=str(payload.get("source_commit_sha", "")),
        source_artifact_sha256=str(payload.get("source_artifact_sha256", "")),
        feature_contract_sha256=str(payload.get("feature_contract_sha256", "")),
        dataset_contract_sha256=str(payload.get("dataset_contract_sha256", "")),
        policy_config=dict(payload.get("policy_config", {}) or {}),
        execution_config=dict(payload.get("execution_config", {}) or {}),
        risk_config=dict(payload.get("risk_config", {}) or {}),
        expected_instruments=tuple(payload.get("expected_instruments", ()) or ()),
        expected_regime_keys=tuple(payload.get("expected_regime_keys", ()) or ()),
        paper_only=payload.get("paper_only", True) is True,
        capital_at_risk=payload.get("capital_at_risk", False) is True,
        broker_exchange_write_enabled=(
            payload.get("broker_exchange_write_enabled", False) is True
        ),
        live_exchange_write_enabled=(
            payload.get("live_exchange_write_enabled", False) is True
        ),
    )


def build_strategy_candidate_manifest(
    *,
    candidate: StrategyCandidate,
    paper_pipeline_config: Mapping[str, Any],
    created_at: str = DEFAULT_STRATEGY_DISCOVERY_CREATED_AT,
) -> StrategyCandidateManifest:
    """Build a deterministic strategy candidate manifest."""

    return StrategyCandidateManifest(
        schema_version=STRATEGY_DISCOVERY_SCHEMA_VERSION,
        phase=STRATEGY_CANDIDATE_MANIFEST_PHASE,
        candidate_id=candidate.candidate_id,
        candidate_sha256=candidate.candidate_sha256,
        candidate_payload=candidate.to_dict(),
        input_artifact_hashes={
            "source_artifact_sha256": candidate.source_artifact_sha256,
            "feature_contract_sha256": candidate.feature_contract_sha256,
            "dataset_contract_sha256": candidate.dataset_contract_sha256,
        },
        expected_phase0_contract="phase0_data_correctness_firewall",
        expected_phase1_policy_contract="phase1_pure_policy_learning",
        expected_phase2_execution_contract="phase2_execution_consistent_pnl",
        paper_pipeline_config=dict(sorted(paper_pipeline_config.items())),
        created_at=created_at,
    )


def raw_candidate_id(payload: Mapping[str, Any], index: int) -> str:
    value = str(payload.get("candidate_id", "")).strip()
    if value:
        return _safe_candidate_dir_name(value)
    return f"invalid_candidate_{index:04d}"


def _safe_candidate_dir_name(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in value)
    return safe or "invalid_candidate"


def _validate_identifier(value: str, field_name: str) -> None:
    if not value.strip():
        raise StrategyDiscoveryError(f"{field_name} is required")
    if value != _safe_candidate_dir_name(value):
        raise StrategyDiscoveryError(
            f"{field_name} must contain only letters, digits, '-' or '_'"
        )
