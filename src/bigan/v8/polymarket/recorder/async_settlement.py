"""Round-scoped pending capture and asynchronous settlement finalization."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.corpus import (
    RAW_CORPUS_FILENAMES,
    PolymarketCorpusBuildConfig,
    build_polymarket_btc_corpus,
)
from bigan.v8.polymarket.corpus.contracts import safety_fields
from bigan.v8.polymarket.recorder.btc_reference import validate_btc_feature_candles
from bigan.v8.polymarket.recorder.contracts import (
    POLYMARKET_REAL_CORPUS_RECORDER_SCHEMA_VERSION,
    PolymarketRealCorpusRecorderConfig,
    empty_raw_payloads,
)
from bigan.v8.polymarket.recorder.operator import (
    _call_provider_stage,
    _provider_safety_failures,
    _raw_market_row,
    _sha256_file,
    _sort_raw_payloads,
    _validate_market_row,
)
from bigan.v8.polymarket.recorder.orderbook_state import (
    validate_market_books,
    validate_trade_rows,
)
from bigan.v8.polymarket.recorder.public_provider import PolymarketRealCorpusPublicProvider
from bigan.v8.polymarket.recorder.resolution import validate_resolution_row
from bigan.v8.polymarket.storage import (
    V8_TRAINING_CORPUS_ROOT,
    export_trainable_corpus,
    round_corpus_id_from_corpus_dir,
)

ASYNC_SETTLEMENT_SCHEMA_VERSION = "bigan-v8-polymarket-async-settlement-v1"
PENDING_CAPTURE_PHASE = "polymarket_pending_round_capture"
PENDING_FINALIZATION_PHASE = "polymarket_pending_round_finalization"


@dataclass(frozen=True, slots=True)
class PendingRoundCaptureResult:
    """Artifacts for a round whose market data is captured before settlement."""

    run_dir: Path
    raw_dir: Path
    artifact_paths: dict[str, Path]
    artifact_hashes: dict[str, str]
    manifest: dict[str, Any]
    report: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PendingRoundFinalizationResult:
    """Artifacts for a pending round finalization attempt."""

    run_dir: Path
    raw_dir: Path
    corpus_dir: Path | None
    exported_training_corpus_dir: Path | None
    artifact_paths: dict[str, Path]
    artifact_hashes: dict[str, str]
    manifest: dict[str, Any]
    report: dict[str, Any]


def capture_polymarket_pending_round(
    config: PolymarketRealCorpusRecorderConfig,
    *,
    public_provider: PolymarketRealCorpusPublicProvider,
) -> PendingRoundCaptureResult:
    """Capture one round's market facts without waiting for delayed settlement."""

    run_dir = config.run_dir
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(
                f"pending capture run_dir already exists: {run_dir}; "
                "set overwrite_existing=True to replace it"
            )
        shutil.rmtree(run_dir)
    config.raw_dir.mkdir(parents=True)

    provider_failures = _provider_safety_failures(public_provider)
    market_candidates: list[dict[str, Any]] = []
    book_candidates: list[dict[str, Any]] = []
    trade_candidates: list[dict[str, Any]] = []
    candle_candidates: list[dict[str, Any]] = []
    if not provider_failures:
        market_candidates = _call_provider_stage(
            provider="polymarket_gamma",
            provider_stage="market_discovery",
            failures=provider_failures,
            callback=lambda: public_provider.market_rows(config),
        )
        book_candidates = _call_provider_stage(
            provider="polymarket_clob",
            provider_stage="orderbook_collection",
            failures=provider_failures,
            callback=lambda: public_provider.orderbook_rows(market_candidates, config),
        )
        trade_candidates = _call_provider_stage(
            provider="polymarket_clob",
            provider_stage="trade_collection",
            failures=provider_failures,
            callback=lambda: public_provider.trade_rows(market_candidates, config),
        )
        candle_candidates = _call_provider_stage(
            provider="btc_reference",
            provider_stage="feature_candle_collection",
            failures=provider_failures,
            callback=lambda: public_provider.btc_feature_candle_rows(market_candidates, config),
        )

    raw_payloads = empty_raw_payloads()
    rejected_rows: list[dict[str, Any]] = list(provider_failures)
    accepted_markets: list[dict[str, Any]] = []
    for market in market_candidates:
        market_reasons = _validate_market_row(market)
        books, book_reasons = validate_market_books(
            market=market,
            book_rows=book_candidates,
            config=config,
        )
        trades, trade_reasons = validate_trade_rows(
            market=market,
            trade_rows=trade_candidates,
        )
        reasons = sorted(set(market_reasons + book_reasons + trade_reasons))
        if reasons:
            rejected_rows.append(_rejected_market(market, reasons))
            continue
        accepted_markets.append(market)
        raw_payloads["raw_polymarket_markets.jsonl"].append(_raw_market_row(market))
        raw_payloads["raw_polymarket_orderbooks.jsonl"].extend(books)
        raw_payloads["raw_polymarket_trades.jsonl"].extend(trades)

    candles, candle_reasons = validate_btc_feature_candles(
        candle_candidates if accepted_markets else []
    )
    if candle_reasons:
        for market in accepted_markets:
            rejected_rows.append(_rejected_market(market, candle_reasons))
        raw_payloads = empty_raw_payloads()
        accepted_markets = []
    else:
        raw_payloads["raw_binance_btcusdt_klines.jsonl"].extend(candles)

    _sort_raw_payloads(raw_payloads)
    _write_raw_files(config.raw_dir, raw_payloads)
    artifact_paths = _pending_capture_paths(run_dir, config.raw_dir)
    _write_jsonl(artifact_paths["pending_round_rejected_rows"], rejected_rows)
    report = _pending_capture_report(
        config=config,
        raw_payloads=raw_payloads,
        rejected_rows=rejected_rows,
        provider_failures=provider_failures,
    )
    manifest = _pending_capture_manifest(
        config=config,
        raw_payloads=raw_payloads,
        report=report,
    )
    _write_json(artifact_paths["pending_round_capture_report"], report)
    _write_json(artifact_paths["pending_round_capture_manifest"], manifest)
    artifact_hashes = {
        name: _sha256_file(path) for name, path in sorted(artifact_paths.items()) if path.exists()
    }
    return PendingRoundCaptureResult(
        run_dir=run_dir,
        raw_dir=config.raw_dir,
        artifact_paths=artifact_paths,
        artifact_hashes=artifact_hashes,
        manifest=manifest,
        report=report,
    )


def finalize_polymarket_pending_round(
    run_dir: Path | str,
    *,
    public_provider: PolymarketRealCorpusPublicProvider,
    destination_root: Path | str = V8_TRAINING_CORPUS_ROOT,
    overwrite_existing: bool = False,
) -> PendingRoundFinalizationResult:
    """Try to add settlement to a pending round and export only after outcome exists."""

    resolved_run_dir = Path(run_dir).expanduser().resolve()
    manifest_path = resolved_run_dir / "pending_round_capture_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing pending capture manifest: {manifest_path}")
    capture_manifest = _read_json(manifest_path)
    config = _config_from_manifest(capture_manifest)
    raw_dir = resolved_run_dir / "raw"
    raw_payloads = _read_raw_payloads(raw_dir)
    market_rows = raw_payloads["raw_polymarket_markets.jsonl"]
    rejected_rows: list[dict[str, Any]] = []
    resolution_rows: list[dict[str, Any]] = []

    if market_rows:
        existing_resolution_candidates = list(raw_payloads["raw_polymarket_resolutions.jsonl"])
        provider_resolution_candidates = _call_provider_stage(
            provider="polymarket_resolution",
            provider_stage="resolution_collection",
            failures=rejected_rows,
            callback=lambda: public_provider.resolution_rows(market_rows, config),
        )
        resolution_candidates = _preferred_resolution_candidates(
            existing_resolution_candidates=existing_resolution_candidates,
            provider_resolution_candidates=provider_resolution_candidates,
        )
        for market in market_rows:
            resolution, reasons = validate_resolution_row(
                market=market,
                resolution_rows=resolution_candidates,
            )
            if reasons:
                rejected_rows.append(_rejected_market(market, reasons))
                continue
            if resolution is not None:
                resolution_rows.append(resolution)
    else:
        rejected_rows.append(
            {
                "provider": "pending_round_capture",
                "provider_stage": "pending_raw_validation",
                "reject_reasons": ["missing_pending_market_rows"],
                "details": "Pending capture has no accepted market rows.",
            }
        )

    raw_payloads["raw_polymarket_resolutions.jsonl"] = resolution_rows
    _sort_raw_payloads(raw_payloads)
    _write_raw_files(raw_dir, raw_payloads)

    corpus_dir = None
    phase2_result = None
    exported_training_corpus_dir = None
    phase2_error = None
    if market_rows and len(resolution_rows) == len(market_rows):
        try:
            phase2_result = build_polymarket_btc_corpus(
                PolymarketCorpusBuildConfig(
                    input_dir=raw_dir,
                    output_dir=resolved_run_dir / "phase2_corpus",
                    created_at=config.created_at,
                    market_families=tuple(config.market_families),  # type: ignore[arg-type]
                    sample_interval_seconds=config.resolved_sampling_policy_seconds(),
                    overwrite_existing=True,
                )
            )
            corpus_dir = phase2_result.output_dir
            round_slug = round_corpus_id_from_corpus_dir(corpus_dir)
            exported_training_corpus_dir = export_trainable_corpus(
                corpus_dir=corpus_dir,
                corpus_id=round_slug,
                destination_root=destination_root,
                overwrite_existing=overwrite_existing,
                provenance={
                    "source": PENDING_FINALIZATION_PHASE,
                    "run_id": config.run_id,
                    "round_slug": round_slug,
                    "corpus_id": round_slug,
                    "round_scoped_export": True,
                    "pending_capture_manifest_path": str(manifest_path),
                    "phase2_corpus_manifest_sha256": phase2_result.artifact_hashes.get(
                        "corpus_manifest"
                    ),
                    "real_historical_corpus_used": True,
                    "manual_live_evidence_eligible": True,
                    "mock_public_data_used": False,
                    "synthetic_public_data_used": False,
                    "synthetic_corpus_used": False,
                    **safety_fields(),
                },
            )
        except Exception as exc:  # noqa: BLE001
            phase2_error = str(exc)

    artifact_paths = _pending_finalization_paths(resolved_run_dir, raw_dir)
    _write_jsonl(artifact_paths["pending_round_finalization_rejected_rows"], rejected_rows)
    report = _pending_finalization_report(
        config=config,
        raw_payloads=raw_payloads,
        rejected_rows=rejected_rows,
        phase2_result=phase2_result,
        phase2_error=phase2_error,
        exported_training_corpus_dir=exported_training_corpus_dir,
    )
    finalization_manifest = _pending_finalization_manifest(
        config=config,
        raw_payloads=raw_payloads,
        report=report,
        phase2_result=phase2_result,
        exported_training_corpus_dir=exported_training_corpus_dir,
    )
    _write_json(artifact_paths["pending_round_finalization_report"], report)
    _write_json(artifact_paths["pending_round_finalization_manifest"], finalization_manifest)
    if phase2_result is not None:
        artifact_paths.update(
            {f"phase2_{name}": path for name, path in phase2_result.artifact_paths.items()}
        )
    artifact_hashes = {
        name: _sha256_file(path) for name, path in sorted(artifact_paths.items()) if path.exists()
    }
    return PendingRoundFinalizationResult(
        run_dir=resolved_run_dir,
        raw_dir=raw_dir,
        corpus_dir=corpus_dir,
        exported_training_corpus_dir=exported_training_corpus_dir,
        artifact_paths=artifact_paths,
        artifact_hashes=artifact_hashes,
        manifest=finalization_manifest,
        report=report,
    )


def _pending_capture_paths(run_dir: Path, raw_dir: Path) -> dict[str, Path]:
    return {
        "pending_round_capture_manifest": run_dir / "pending_round_capture_manifest.json",
        "pending_round_capture_report": run_dir / "pending_round_capture_report.json",
        "pending_round_rejected_rows": run_dir / "pending_round_rejected_rows.jsonl",
        **{filename: raw_dir / filename for filename in RAW_CORPUS_FILENAMES},
    }


def _pending_finalization_paths(run_dir: Path, raw_dir: Path) -> dict[str, Path]:
    return {
        "pending_round_finalization_manifest": (
            run_dir / "pending_round_finalization_manifest.json"
        ),
        "pending_round_finalization_report": run_dir / "pending_round_finalization_report.json",
        "pending_round_finalization_rejected_rows": (
            run_dir / "pending_round_finalization_rejected_rows.jsonl"
        ),
        **{filename: raw_dir / filename for filename in RAW_CORPUS_FILENAMES},
    }


def _preferred_resolution_candidates(
    *,
    existing_resolution_candidates: list[dict[str, Any]],
    provider_resolution_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    provider_market_ids = {
        str(row.get("market_id") or "")
        for row in provider_resolution_candidates
        if str(row.get("market_id") or "")
    }
    return [
        *provider_resolution_candidates,
        *[
            row
            for row in existing_resolution_candidates
            if str(row.get("market_id") or "") not in provider_market_ids
        ],
    ]


def _pending_capture_report(
    *,
    config: PolymarketRealCorpusRecorderConfig,
    raw_payloads: dict[str, list[dict[str, Any]]],
    rejected_rows: list[dict[str, Any]],
    provider_failures: list[dict[str, Any]],
) -> dict[str, Any]:
    market_count = len(raw_payloads["raw_polymarket_markets.jsonl"])
    reject_counts = _reject_counts(rejected_rows)
    return {
        "schema_version": ASYNC_SETTLEMENT_SCHEMA_VERSION,
        "phase": PENDING_CAPTURE_PHASE,
        "run_id": config.run_id,
        "market_families": list(config.market_families),
        "pending_resolution": market_count > 0,
        "capture_status": (
            "blocked_fail_closed"
            if provider_failures or market_count == 0
            else "pending_resolution"
        ),
        "resolution_provider_called": False,
        "raw_polymarket_market_count": market_count,
        "raw_orderbook_row_count": len(raw_payloads["raw_polymarket_orderbooks.jsonl"]),
        "raw_trade_row_count": len(raw_payloads["raw_polymarket_trades.jsonl"]),
        "raw_btc_candle_row_count": len(raw_payloads["raw_binance_btcusdt_klines.jsonl"]),
        "raw_resolution_count": 0,
        "rejected_row_count": len(rejected_rows),
        "reject_reason_counts": reject_counts,
        "training_eligible": False,
        "phase2_corpus_built": False,
        "exported_training_corpus_dir": None,
        "public_collection_reason_codes": sorted(
            {reason for row in provider_failures for reason in row.get("reject_reasons", [])}
        ),
        "mock_public_data_used": False,
        "synthetic_public_data_used": False,
        "synthetic_corpus_used": False,
        "real_historical_corpus_used": False,
        **safety_fields(),
    }


def _pending_capture_manifest(
    *,
    config: PolymarketRealCorpusRecorderConfig,
    raw_payloads: dict[str, list[dict[str, Any]]],
    report: dict[str, Any],
) -> dict[str, Any]:
    raw_paths = {filename: config.raw_dir / filename for filename in RAW_CORPUS_FILENAMES}
    return {
        "schema_version": ASYNC_SETTLEMENT_SCHEMA_VERSION,
        "phase": PENDING_CAPTURE_PHASE,
        "recorder_schema_version": POLYMARKET_REAL_CORPUS_RECORDER_SCHEMA_VERSION,
        "run_id": config.run_id,
        "config": config.to_dict(),
        "raw_artifact_hashes": {
            filename: _sha256_file(path) for filename, path in raw_paths.items() if path.exists()
        },
        "raw_artifact_row_counts": {
            filename: len(raw_payloads[filename]) for filename in RAW_CORPUS_FILENAMES
        },
        "pending_resolution": report["pending_resolution"],
        "capture_status": report["capture_status"],
        "resolution_provider_called": False,
        **safety_fields(),
    }


def _pending_finalization_report(
    *,
    config: PolymarketRealCorpusRecorderConfig,
    raw_payloads: dict[str, list[dict[str, Any]]],
    rejected_rows: list[dict[str, Any]],
    phase2_result: Any,
    phase2_error: str | None,
    exported_training_corpus_dir: Path | None,
) -> dict[str, Any]:
    market_count = len(raw_payloads["raw_polymarket_markets.jsonl"])
    resolution_count = len(raw_payloads["raw_polymarket_resolutions.jsonl"])
    exported = phase2_result is not None and exported_training_corpus_dir is not None
    if exported:
        status = "exported"
    elif market_count > 0 and resolution_count < market_count:
        status = "pending_resolution"
    else:
        status = "blocked_fail_closed"
    return {
        "schema_version": ASYNC_SETTLEMENT_SCHEMA_VERSION,
        "phase": PENDING_FINALIZATION_PHASE,
        "run_id": config.run_id,
        "finalization_status": status,
        "pending_resolution": status == "pending_resolution",
        "raw_polymarket_market_count": market_count,
        "raw_orderbook_row_count": len(raw_payloads["raw_polymarket_orderbooks.jsonl"]),
        "raw_trade_row_count": len(raw_payloads["raw_polymarket_trades.jsonl"]),
        "raw_btc_candle_row_count": len(raw_payloads["raw_binance_btcusdt_klines.jsonl"]),
        "raw_resolution_count": resolution_count,
        "rejected_row_count": len(rejected_rows),
        "reject_reason_counts": _reject_counts(rejected_rows),
        "training_eligible": exported,
        "phase2_corpus_built": phase2_result is not None,
        "phase2_error": phase2_error,
        "phase2_corpus_manifest_sha256": (
            None
            if phase2_result is None
            else phase2_result.artifact_hashes.get("corpus_manifest")
        ),
        "exported_training_corpus_dir": (
            None if exported_training_corpus_dir is None else str(exported_training_corpus_dir)
        ),
        "mock_public_data_used": False,
        "synthetic_public_data_used": False,
        "synthetic_corpus_used": False,
        "real_historical_corpus_used": exported,
        "manual_live_evidence_eligible": exported,
        **safety_fields(),
    }


def _pending_finalization_manifest(
    *,
    config: PolymarketRealCorpusRecorderConfig,
    raw_payloads: dict[str, list[dict[str, Any]]],
    report: dict[str, Any],
    phase2_result: Any,
    exported_training_corpus_dir: Path | None,
) -> dict[str, Any]:
    raw_paths = {filename: config.raw_dir / filename for filename in RAW_CORPUS_FILENAMES}
    return {
        "schema_version": ASYNC_SETTLEMENT_SCHEMA_VERSION,
        "phase": PENDING_FINALIZATION_PHASE,
        "run_id": config.run_id,
        "config": config.to_dict(),
        "raw_artifact_hashes": {
            filename: _sha256_file(path) for filename, path in raw_paths.items() if path.exists()
        },
        "raw_artifact_row_counts": {
            filename: len(raw_payloads[filename]) for filename in RAW_CORPUS_FILENAMES
        },
        "finalization_status": report["finalization_status"],
        "pending_resolution": report["pending_resolution"],
        "phase2_corpus_built": phase2_result is not None,
        "phase2_corpus_dir": None if phase2_result is None else str(phase2_result.output_dir),
        "phase2_corpus_manifest_sha256": report["phase2_corpus_manifest_sha256"],
        "exported_training_corpus_dir": (
            None if exported_training_corpus_dir is None else str(exported_training_corpus_dir)
        ),
        **safety_fields(),
    }


def _config_from_manifest(manifest: dict[str, Any]) -> PolymarketRealCorpusRecorderConfig:
    config = dict(manifest["config"])
    config["output_dir"] = Path(config["output_dir"])
    config["market_families"] = tuple(config["market_families"])
    return PolymarketRealCorpusRecorderConfig(**config)


def _read_raw_payloads(raw_dir: Path) -> dict[str, list[dict[str, Any]]]:
    return {filename: _read_jsonl(raw_dir / filename) for filename in RAW_CORPUS_FILENAMES}


def _write_raw_files(raw_dir: Path, raw_payloads: dict[str, list[dict[str, Any]]]) -> None:
    for filename in RAW_CORPUS_FILENAMES:
        _write_jsonl(raw_dir / filename, raw_payloads[filename])


def _rejected_market(market: dict[str, Any], reasons: list[str]) -> dict[str, Any]:
    return {
        "market_id": market.get("market_id"),
        "slug": market.get("slug"),
        "market_family": market.get("market_family"),
        "reject_reasons": sorted(set(reasons)),
    }


def _reject_counts(rejected_rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for row in rejected_rows:
        for reason in row.get("reject_reasons", []):
            counts[str(reason)] += 1
    return dict(sorted(counts.items()))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )
