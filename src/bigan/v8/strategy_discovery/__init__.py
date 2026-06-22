"""Strategy Discovery integration for v8 paper-only validation."""

from bigan.v8.strategy_discovery.contracts import (
    DEFAULT_STRATEGY_DISCOVERY_CREATED_AT,
    STRATEGY_CANDIDATE_MANIFEST_PHASE,
    STRATEGY_DISCOVERY_SCHEMA_VERSION,
    CandidateStatus,
    StrategyCandidate,
    StrategyCandidateManifest,
    StrategyDiscoveryError,
    build_strategy_candidate_manifest,
    strategy_candidate_from_mapping,
)
from bigan.v8.strategy_discovery.paper_integration import (
    STRATEGY_DISCOVERY_PAPER_INTEGRATION_PHASE,
    StrategyCandidateReplayBatchResult,
    StrategyCandidateReplayConfig,
    load_strategy_candidates_jsonl,
    run_strategy_candidate_replay_batch,
)
from bigan.v8.strategy_discovery.registry import (
    StrategyCandidateRegistry,
    StrategyCandidateRegistryEntry,
    build_strategy_candidate_registry,
)

__all__ = [
    "DEFAULT_STRATEGY_DISCOVERY_CREATED_AT",
    "STRATEGY_CANDIDATE_MANIFEST_PHASE",
    "STRATEGY_DISCOVERY_PAPER_INTEGRATION_PHASE",
    "STRATEGY_DISCOVERY_SCHEMA_VERSION",
    "CandidateStatus",
    "StrategyCandidate",
    "StrategyCandidateManifest",
    "StrategyCandidateRegistry",
    "StrategyCandidateRegistryEntry",
    "StrategyCandidateReplayBatchResult",
    "StrategyCandidateReplayConfig",
    "StrategyDiscoveryError",
    "build_strategy_candidate_manifest",
    "build_strategy_candidate_registry",
    "load_strategy_candidates_jsonl",
    "run_strategy_candidate_replay_batch",
    "strategy_candidate_from_mapping",
]
