"""Candidate registry records for strategy discovery paper replay."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from bigan.v8.paper.contracts import canonical_payload_sha256
from bigan.v8.strategy_discovery.contracts import (
    DEFAULT_STRATEGY_DISCOVERY_CREATED_AT,
    STRATEGY_DISCOVERY_SCHEMA_VERSION,
    CandidateStatus,
)


@dataclass(frozen=True, slots=True)
class StrategyCandidateRegistryEntry:
    """One candidate outcome retained in the replay audit registry."""

    candidate_id: str
    status: CandidateStatus
    candidate_sha256: str | None
    candidate_manifest_sha256: str | None
    candidate_manifest_path: str | None
    candidate_replay_summary_path: str
    operator_recommendation: str | None
    phase6_deployment_status: str | None
    critical_alert_count: int
    warning_alert_count: int
    artifact_hashes: dict[str, str]
    reason_codes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class StrategyCandidateRegistry:
    """Batch-level deterministic candidate registry."""

    schema_version: str
    batch_id: str
    candidate_count: int
    entries: list[dict[str, Any]]
    paper_only: bool
    capital_at_risk: bool
    broker_exchange_write_enabled: bool
    live_exchange_write_enabled: bool
    created_at: str

    @property
    def registry_sha256(self) -> str:
        return canonical_payload_sha256(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = asdict(self)
        if include_hash:
            payload["registry_sha256"] = self.registry_sha256
        return payload


def build_strategy_candidate_registry(
    *,
    batch_id: str,
    entries: list[StrategyCandidateRegistryEntry],
    created_at: str = DEFAULT_STRATEGY_DISCOVERY_CREATED_AT,
) -> StrategyCandidateRegistry:
    """Build a deterministic batch candidate registry."""

    entry_payloads = [
        entry.to_dict()
        for entry in sorted(entries, key=lambda item: item.candidate_id)
    ]
    return StrategyCandidateRegistry(
        schema_version=STRATEGY_DISCOVERY_SCHEMA_VERSION,
        batch_id=batch_id,
        candidate_count=len(entry_payloads),
        entries=entry_payloads,
        paper_only=True,
        capital_at_risk=False,
        broker_exchange_write_enabled=False,
        live_exchange_write_enabled=False,
        created_at=created_at,
    )
