"""Aggregate round-scoped Polymarket corpora into a retraining bundle."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.corpus.builder import build_polymarket_btc_corpus
from bigan.v8.polymarket.corpus.contracts import (
    DEFAULT_CORPUS_CREATED_AT,
    CorpusMarketFamily,
    PolymarketCorpusBuildConfig,
    PolymarketCorpusBuildResult,
    safety_fields,
)

POLYMARKET_REAL_CORPUS_AGGREGATE_SCHEMA_VERSION = (
    "bigan-v8-polymarket-real-corpus-aggregate-v1"
)
SPARSE_THEORETICAL_TRAINING_EXCLUSION_REASON = (
    "sparse_theoretical_sell_before_close_training_exclusion"
)

_NORMALIZED_TO_RAW_FILENAMES = {
    "polymarket_market_metadata.jsonl": "raw_polymarket_markets.jsonl",
    "polymarket_token_book_snapshots.jsonl": "raw_polymarket_orderbooks.jsonl",
    "polymarket_token_trades.jsonl": "raw_polymarket_trades.jsonl",
    "polymarket_btc_reference_candles.jsonl": "raw_binance_btcusdt_klines.jsonl",
    "polymarket_resolution_events.jsonl": "raw_polymarket_resolutions.jsonl",
}
_RAW_ROW_COUNT_KEYS = {
    "raw_polymarket_markets.jsonl": "market_rows",
    "raw_polymarket_orderbooks.jsonl": "book_snapshot_rows",
    "raw_polymarket_trades.jsonl": "trade_rows",
    "raw_binance_btcusdt_klines.jsonl": "btc_reference_candle_rows",
    "raw_polymarket_resolutions.jsonl": "resolution_rows",
}


@dataclass(frozen=True, slots=True)
class PolymarketRealCorpusAggregateConfig:
    """Configuration for a deterministic real-corpus aggregate rebuild."""

    source_root: Path | str
    output_dir: Path | str
    run_id: str
    created_at: str = DEFAULT_CORPUS_CREATED_AT
    market_families: tuple[CorpusMarketFamily, ...] = ("btc_updown_5m",)
    sample_interval_seconds: dict[str, int] | None = None
    sell_before_close_entry_notional: float = 1.0
    sell_before_close_min_exit_notional: float = 1.0
    exclude_sparse_theoretical_sell_before_close: bool = True
    overwrite_existing: bool = False
    paper_only: bool = True
    capital_at_risk: bool = False
    broker_exchange_write_enabled: bool = False
    live_exchange_write_enabled: bool = False
    polymarket_write_enabled: bool = False
    wallet_signing_enabled: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.source_root, Path):
            object.__setattr__(self, "source_root", Path(self.source_root))
        if not isinstance(self.output_dir, Path):
            object.__setattr__(self, "output_dir", Path(self.output_dir))
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if not self.market_families:
            raise ValueError("market_families must not be empty")
        for field_name, expected in safety_fields().items():
            if getattr(self, field_name) is not expected:
                raise ValueError(f"{field_name} must be {str(expected).lower()}")

    @property
    def run_dir(self) -> Path:
        return self.output_dir.expanduser().resolve() / self.run_id

    @property
    def corpus_dir(self) -> Path:
        return self.run_dir / "corpus"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_root": str(self.source_root),
            "output_dir": str(self.output_dir),
            "run_id": self.run_id,
            "created_at": self.created_at,
            "market_families": list(self.market_families),
            "sample_interval_seconds": self.resolved_sample_intervals(),
            "sell_before_close_entry_notional": self.sell_before_close_entry_notional,
            "sell_before_close_min_exit_notional": (
                self.sell_before_close_min_exit_notional
            ),
            "exclude_sparse_theoretical_sell_before_close": (
                self.exclude_sparse_theoretical_sell_before_close
            ),
            "overwrite_existing": self.overwrite_existing,
            **safety_fields(),
        }

    def resolved_sample_intervals(self) -> dict[str, int]:
        default = {
            "btc_updown_5m": 60,
            "btc_updown_15m": 300,
            "btc_updown_1h": 900,
        }
        if self.sample_interval_seconds:
            default.update(
                {
                    str(key): int(value)
                    for key, value in self.sample_interval_seconds.items()
                }
            )
        return default


@dataclass(frozen=True, slots=True)
class PolymarketRealCorpusAggregateResult:
    """Output handles for a real-corpus aggregate rebuild."""

    run_dir: Path
    corpus_dir: Path
    artifact_paths: dict[str, Path]
    report: dict[str, Any]
    manifest: dict[str, Any]
    aggregate_source_corpora: dict[str, Any]
    phase2_result: PolymarketCorpusBuildResult


def build_polymarket_real_corpus_aggregate(
    config: PolymarketRealCorpusAggregateConfig,
) -> PolymarketRealCorpusAggregateResult:
    """Build a gate-compatible aggregate corpus from round-scoped corpora."""

    run_dir = config.run_dir
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(
                f"aggregate run_dir already exists: {run_dir}; "
                "set overwrite_existing=True to replace it"
            )
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    source_infos = _discover_source_corpora(config.source_root.expanduser().resolve())
    if not source_infos:
        raise ValueError("no source corpora found")
    excluded_by_reason = _probe_sparse_theoretical_exclusions(
        source_infos=source_infos,
        config=config,
    )
    included_sources = [
        info for info in source_infos if info["corpus_dir"] not in excluded_by_reason
    ]
    excluded_sources = [
        {
            **info,
            "excluded_reason": excluded_by_reason[info["corpus_dir"]],
        }
        for info in source_infos
        if info["corpus_dir"] in excluded_by_reason
    ]
    if not included_sources:
        raise ValueError("all source corpora were excluded")

    phase2_result = _build_from_sources(
        source_infos=included_sources,
        output_dir=config.corpus_dir,
        config=config,
    )
    label_report = _read_json(
        phase2_result.output_dir / "sell_before_close_label_redesign_report.json"
    )
    aggregate_source_corpora = _aggregate_source_corpora_payload(
        config=config,
        included_sources=included_sources,
        excluded_sources=excluded_sources,
    )
    artifact_paths = {
        "aggregate_source_corpora": run_dir / "aggregate_source_corpora.json",
        "real_corpus_recorder_manifest": run_dir / "real_corpus_recorder_manifest.json",
        "real_corpus_recorder_report": run_dir / "real_corpus_recorder_report.json",
        "aggregate_summary": run_dir / "polymarket_real_corpus_aggregate_summary.json",
    }
    _write_json(artifact_paths["aggregate_source_corpora"], aggregate_source_corpora)
    aggregate_source_sha = _sha256_file(artifact_paths["aggregate_source_corpora"])
    report = _recorder_report(
        config=config,
        phase2_result=phase2_result,
        label_report=label_report,
        included_sources=included_sources,
        excluded_sources=excluded_sources,
        aggregate_source_manifest_sha256=aggregate_source_sha,
    )
    manifest = {
        **report,
        "config": config.to_dict(),
    }
    _write_json(artifact_paths["real_corpus_recorder_report"], report)
    _write_json(artifact_paths["real_corpus_recorder_manifest"], manifest)
    summary = {
        "run_id": config.run_id,
        "phase2_corpus_dir": str(phase2_result.output_dir),
        "phase2_corpus_manifest_sha256": report["phase2_corpus_manifest_sha256"],
        "included_source_corpus_count": len(included_sources),
        "excluded_source_corpus_count": len(excluded_sources),
        "excluded_reason_counts": report["excluded_reason_counts"],
        "sell_before_close_label_gate_passed": report[
            "sell_before_close_label_gate_passed"
        ],
        "sell_before_close_execution_class_counts": report[
            "sell_before_close_execution_class_counts"
        ],
        "real_historical_training_eligible": report[
            "real_historical_training_eligible"
        ],
        **safety_fields(),
    }
    _write_json(artifact_paths["aggregate_summary"], summary)
    return PolymarketRealCorpusAggregateResult(
        run_dir=run_dir,
        corpus_dir=phase2_result.output_dir,
        artifact_paths=artifact_paths,
        report=report,
        manifest=manifest,
        aggregate_source_corpora=aggregate_source_corpora,
        phase2_result=phase2_result,
    )


def _discover_source_corpora(source_root: Path) -> list[dict[str, Any]]:
    source_infos = []
    for corpus_dir in sorted(source_root.iterdir()):
        if not corpus_dir.is_dir() or corpus_dir.name.startswith("."):
            continue
        manifest_path = corpus_dir / "polymarket_corpus_manifest.json"
        market_path = corpus_dir / "polymarket_market_metadata.jsonl"
        if not manifest_path.exists() or not market_path.exists():
            continue
        manifest = _read_json(manifest_path)
        markets = _read_jsonl(market_path)
        source_infos.append(
            {
                "corpus_dir": str(corpus_dir),
                "round_id": corpus_dir.name,
                "corpus_manifest_sha256": _sha256_file(manifest_path),
                "market_count": int(manifest.get("market_count", len(markets))),
                "feature_row_count": int(manifest.get("feature_row_count", 0)),
                "label_row_count": int(manifest.get("label_row_count", 0)),
                "market_ids": [str(row["market_id"]) for row in markets],
                "slugs": [str(row["slug"]) for row in markets],
                **safety_fields(),
            }
        )
    return source_infos


def _probe_sparse_theoretical_exclusions(
    *,
    source_infos: list[dict[str, Any]],
    config: PolymarketRealCorpusAggregateConfig,
) -> dict[str, str]:
    if not config.exclude_sparse_theoretical_sell_before_close:
        return {}
    with tempfile.TemporaryDirectory(prefix="v8-polymarket-aggregate-probe-") as tmp:
        probe_dir = Path(tmp) / "probe_corpus"
        probe_result = _build_from_sources(
            source_infos=source_infos,
            output_dir=probe_dir,
            config=config,
        )
        label_path = probe_result.output_dir / "polymarket_label_rows.jsonl"
        sparse_market_ids = {
            str(row["market_id"])
            for row in _read_jsonl(label_path)
            if row.get("sell_before_close_execution_class")
            == "sparse_theoretical_sell_before_close"
        }
    if not sparse_market_ids:
        return {}
    excluded = {}
    for info in source_infos:
        if sparse_market_ids & set(info["market_ids"]):
            excluded[info["corpus_dir"]] = SPARSE_THEORETICAL_TRAINING_EXCLUSION_REASON
    return excluded


def _build_from_sources(
    *,
    source_infos: list[dict[str, Any]],
    output_dir: Path,
    config: PolymarketRealCorpusAggregateConfig,
) -> PolymarketCorpusBuildResult:
    with tempfile.TemporaryDirectory(prefix="v8-polymarket-aggregate-raw-") as tmp:
        raw_dir = Path(tmp)
        for normalized_name, raw_name in _NORMALIZED_TO_RAW_FILENAMES.items():
            raw_path = raw_dir / raw_name
            with raw_path.open("w", encoding="utf-8") as output:
                for info in source_infos:
                    source_path = Path(info["corpus_dir"]) / normalized_name
                    if not source_path.exists():
                        raise FileNotFoundError(f"missing source artifact: {source_path}")
                    with source_path.open("r", encoding="utf-8") as handle:
                        for line in handle:
                            if line.strip():
                                output.write(line.rstrip("\n") + "\n")
        return build_polymarket_btc_corpus(
            PolymarketCorpusBuildConfig(
                input_dir=raw_dir,
                output_dir=output_dir,
                created_at=config.created_at,
                market_families=config.market_families,
                sample_interval_seconds=config.resolved_sample_intervals(),
                sell_before_close_entry_notional=(
                    config.sell_before_close_entry_notional
                ),
                sell_before_close_min_exit_notional=(
                    config.sell_before_close_min_exit_notional
                ),
                overwrite_existing=True,
            )
        )


def _aggregate_source_corpora_payload(
    *,
    config: PolymarketRealCorpusAggregateConfig,
    included_sources: list[dict[str, Any]],
    excluded_sources: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": POLYMARKET_REAL_CORPUS_AGGREGATE_SCHEMA_VERSION,
        "run_id": config.run_id,
        "source_root": str(config.source_root),
        "included": included_sources,
        "excluded": excluded_sources,
        "excluded_reason_counts": dict(
            sorted(Counter(row["excluded_reason"] for row in excluded_sources).items())
        ),
        **safety_fields(),
    }


def _recorder_report(
    *,
    config: PolymarketRealCorpusAggregateConfig,
    phase2_result: PolymarketCorpusBuildResult,
    label_report: dict[str, Any],
    included_sources: list[dict[str, Any]],
    excluded_sources: list[dict[str, Any]],
    aggregate_source_manifest_sha256: str,
) -> dict[str, Any]:
    manifest = phase2_result.manifest
    corpus_dir = phase2_result.output_dir
    raw_row_counts = _raw_row_counts(corpus_dir)
    raw_row_counts.update(
        {
            "feature_rows": int(manifest["feature_row_count"]),
            "label_rows": int(manifest["label_row_count"]),
            "source_corpus_count": len(included_sources),
            "excluded_source_corpus_count": len(excluded_sources),
        }
    )
    excluded_reason_counts = dict(
        sorted(Counter(row["excluded_reason"] for row in excluded_sources).items())
    )
    label_gate_passed = bool(label_report["label_gate_passed"])
    reason_codes = (
        []
        if label_gate_passed
        else ["sell_before_close_label_redesign_gate_failed"]
    )
    reason_codes.extend(label_report.get("label_gate_reason_codes", []))
    excluded_market_ids = sorted(
        {market_id for row in excluded_sources for market_id in row["market_ids"]}
    )
    excluded_slugs = sorted({slug for row in excluded_sources for slug in row["slugs"]})
    return {
        "schema_version": "bigan-v8-polymarket-real-corpus-recorder-v1",
        "phase": "polymarket_real_corpus_recorder",
        "run_id": config.run_id,
        "created_at": config.created_at,
        "started_at": config.created_at,
        "ended_at": config.created_at,
        "wall_clock_duration_seconds": 0.0,
        "source_root": str(config.source_root),
        "requested_live_public_collection": False,
        "public_collection_status": "completed",
        "public_collection_reason_codes": [],
        "live_polymarket_data": True,
        "live_polymarket_data_read": True,
        "live_btc_reference_data": True,
        "live_btc_reference_data_read": True,
        "mock_public_data_used": False,
        "synthetic_public_data_used": False,
        "synthetic_corpus_used": False,
        "fixture_corpus_used": False,
        "real_historical_corpus_used": True,
        "phase2_corpus_built": True,
        "phase2_corpus_build_eligible": True,
        "phase2_corpus_dir": str(corpus_dir),
        "phase2_corpus_manifest_sha256": phase2_result.artifact_hashes[
            "corpus_manifest"
        ],
        "phase2_train_shadow_split_sha256": phase2_result.artifact_hashes[
            "train_shadow_split"
        ],
        "raw_artifact_hashes": manifest["raw_artifact_hashes"],
        "raw_artifact_row_counts": raw_row_counts,
        "raw_polymarket_market_count": raw_row_counts["market_rows"],
        "raw_orderbook_row_count": raw_row_counts["book_snapshot_rows"],
        "raw_trade_row_count": raw_row_counts["trade_rows"],
        "raw_btc_candle_row_count": raw_row_counts["btc_reference_candle_rows"],
        "raw_resolution_count": raw_row_counts["resolution_rows"],
        "feature_row_count": int(manifest["feature_row_count"]),
        "label_row_count": int(manifest["label_row_count"]),
        "target_label_count": int(manifest["label_row_count"]),
        "target_outcome_counts": {},
        "reject_reason_counts": {},
        "rejected_row_count": 0,
        "included_source_corpus_count": len(included_sources),
        "excluded_source_corpus_count": len(excluded_sources),
        "source_corpora_excluded_count": len(excluded_sources),
        "excluded_market_count": len(excluded_market_ids),
        "excluded_market_ids": excluded_market_ids,
        "excluded_slugs": excluded_slugs,
        "excluded_reason_counts": excluded_reason_counts,
        "aggregate_source_manifest_sha256": aggregate_source_manifest_sha256,
        "excluded_source_manifest_sha256": hashlib.sha256(
            json.dumps(excluded_sources, sort_keys=True).encode()
        ).hexdigest(),
        "market_families": list(config.market_families),
        "sampling_policy": config.resolved_sample_intervals(),
        "training_eligible": label_gate_passed,
        "real_historical_training_eligible": label_gate_passed,
        "manual_live_evidence_eligible": True,
        "reason_codes": reason_codes,
        "sell_before_close_label_schema_version": label_report[
            "sell_before_close_label_schema_version"
        ],
        "sell_before_close_label_gate_passed": label_gate_passed,
        "sell_before_close_label_gate_reason_codes": label_report.get(
            "label_gate_reason_codes", []
        ),
        "sell_before_close_execution_class_counts": label_report[
            "sell_before_close_execution_class_counts"
        ],
        "sparse_theoretical_sell_before_close_count": label_report.get(
            "sparse_theoretical_sell_before_close_count", 0
        ),
        "theoretical_sell_before_close_count": label_report[
            "theoretical_sell_before_close_count"
        ],
        "sell_before_close_entry_notional": label_report[
            "sell_before_close_entry_notional"
        ],
        "sell_before_close_min_exit_notional": label_report[
            "sell_before_close_min_exit_notional"
        ],
        "min_exit_notional_source": label_report["min_exit_notional_source"],
        "min_exit_notional_to_entry_notional_ratio": label_report[
            "min_exit_notional_to_entry_notional_ratio"
        ],
        "near_miss_theoretical_count": label_report["near_miss_theoretical_count"],
        "near_miss_threshold": label_report["near_miss_threshold"],
        **safety_fields(),
    }


def _raw_row_counts(corpus_dir: Path) -> dict[str, int]:
    return {
        _RAW_ROW_COUNT_KEYS[raw_name]: _count_jsonl(corpus_dir / normalized_name)
        for normalized_name, raw_name in _NORMALIZED_TO_RAW_FILENAMES.items()
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())
