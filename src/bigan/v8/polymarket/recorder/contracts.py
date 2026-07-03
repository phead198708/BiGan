"""Contracts for the v8 Polymarket raw corpus recorder."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.corpus import (
    BTC_UPDOWN_MARKET_HORIZONS_MS,
    DEFAULT_CORPUS_CREATED_AT,
    RAW_CORPUS_FILENAMES,
    PolymarketCorpusBuildResult,
)
from bigan.v8.polymarket.corpus.contracts import (
    safety_fields,
)

POLYMARKET_REAL_CORPUS_RECORDER_SCHEMA_VERSION = "bigan-v8-polymarket-real-corpus-recorder-v1"
POLYMARKET_REAL_CORPUS_RECORDER_PHASE = "polymarket_real_corpus_recorder"
DEFAULT_RECORDER_STARTED_AT = "1970-01-01T00:00:00Z"
DEFAULT_RECORDER_ENDED_AT = "1970-01-01T00:00:00Z"
DEFAULT_OFFICIAL_SETTLEMENT_REFERENCE_SOURCE = "polymarket_official_btc_usd_reference"
DEFAULT_BTC_FEATURE_CANDLE_SOURCE = "binance_btcusdt"

DEFAULT_SAMPLING_POLICY_SECONDS: dict[str, int] = {
    "btc_updown_5m": 60,
    "btc_updown_15m": 300,
    "btc_updown_1h": 900,
}


@dataclass(frozen=True, slots=True)
class PolymarketRealCorpusRecorderConfig:
    """Configuration for one read-only raw corpus recording run."""

    run_id: str
    output_dir: Path | str
    created_at: str = DEFAULT_CORPUS_CREATED_AT
    started_at: str = DEFAULT_RECORDER_STARTED_AT
    ended_at: str = DEFAULT_RECORDER_ENDED_AT
    market_families: tuple[str, ...] = tuple(BTC_UPDOWN_MARKET_HORIZONS_MS)
    sampling_policy_seconds: dict[str, int] | None = None
    build_phase2_corpus: bool = True
    mock_public_data: bool = True
    overwrite_existing: bool = False
    official_settlement_reference_source: str = DEFAULT_OFFICIAL_SETTLEMENT_REFERENCE_SOURCE
    btc_feature_candle_source: str = DEFAULT_BTC_FEATURE_CANDLE_SOURCE
    candle_timeframe_ms: int = 60_000
    inject_missing_down_book: bool = False
    inject_unknown_token_book: bool = False
    inject_stale_book: bool = False
    inject_missing_resolution: bool = False
    inject_missing_reference_source: bool = False
    paper_only: bool = True
    capital_at_risk: bool = False
    broker_exchange_write_enabled: bool = False
    live_exchange_write_enabled: bool = False
    polymarket_write_enabled: bool = False
    wallet_signing_enabled: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if not isinstance(self.output_dir, Path):
            object.__setattr__(self, "output_dir", Path(self.output_dir))
        if not self.created_at or not self.started_at or not self.ended_at:
            raise ValueError("created_at, started_at, and ended_at are required")
        unsupported = set(self.market_families) - set(BTC_UPDOWN_MARKET_HORIZONS_MS)
        if unsupported:
            raise ValueError("unsupported market families: " + ", ".join(sorted(unsupported)))
        if not self.market_families:
            raise ValueError("market_families must not be empty")
        if not self.official_settlement_reference_source.strip():
            raise ValueError("official_settlement_reference_source is required")
        if not self.btc_feature_candle_source.strip():
            raise ValueError("btc_feature_candle_source is required")
        if self.candle_timeframe_ms <= 0:
            raise ValueError("candle_timeframe_ms must be positive")
        for family, seconds in self.resolved_sampling_policy_seconds().items():
            if family not in BTC_UPDOWN_MARKET_HORIZONS_MS:
                raise ValueError(f"unsupported sampling family: {family}")
            if seconds <= 0:
                raise ValueError("sampling intervals must be positive")
        for field_name, expected in safety_fields().items():
            if getattr(self, field_name) is not expected:
                raise ValueError(f"{field_name} must be {str(expected).lower()}")

    @property
    def run_dir(self) -> Path:
        return self.output_dir.expanduser().resolve() / self.run_id

    @property
    def raw_dir(self) -> Path:
        return self.run_dir / "raw"

    @property
    def corpus_dir(self) -> Path:
        return self.run_dir / "phase2_corpus"

    def resolved_sampling_policy_seconds(self) -> dict[str, int]:
        policy = dict(DEFAULT_SAMPLING_POLICY_SECONDS)
        if self.sampling_policy_seconds:
            policy.update(
                {str(key): int(value) for key, value in self.sampling_policy_seconds.items()}
            )
        return {family: policy[family] for family in self.market_families}

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["market_families"] = list(self.market_families)
        payload["sampling_policy_seconds"] = self.resolved_sampling_policy_seconds()
        return payload


@dataclass(frozen=True, slots=True)
class PolymarketRealCorpusRecorderResult:
    """Output handles for one raw corpus recording run."""

    run_dir: Path
    raw_dir: Path
    corpus_dir: Path | None
    artifact_paths: dict[str, Path]
    artifact_hashes: dict[str, str]
    manifest: dict[str, Any]
    report: dict[str, Any]
    phase2_result: PolymarketCorpusBuildResult | None


def empty_raw_payloads() -> dict[str, list[dict[str, Any]]]:
    return {filename: [] for filename in RAW_CORPUS_FILENAMES}
