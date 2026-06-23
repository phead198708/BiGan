"""Operator for recording Polymarket market facts into v8 Phase 2 raw corpus files."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.corpus import (
    RAW_CORPUS_FILENAMES,
    PolymarketCorpusBuildConfig,
    build_polymarket_btc_corpus,
)
from bigan.v8.polymarket.corpus.contracts import (
    safety_fields,
)
from bigan.v8.polymarket.recorder.btc_reference import (
    mock_btc_feature_candle_rows,
    validate_btc_feature_candles,
)
from bigan.v8.polymarket.recorder.contracts import (
    POLYMARKET_REAL_CORPUS_RECORDER_PHASE,
    POLYMARKET_REAL_CORPUS_RECORDER_SCHEMA_VERSION,
    PolymarketRealCorpusRecorderConfig,
    PolymarketRealCorpusRecorderResult,
    empty_raw_payloads,
)
from bigan.v8.polymarket.recorder.market_discovery import discover_mock_market_rows
from bigan.v8.polymarket.recorder.orderbook_state import (
    mock_orderbook_rows,
    mock_trade_rows,
    validate_market_books,
    validate_trade_rows,
)
from bigan.v8.polymarket.recorder.resolution import (
    mock_resolution_rows,
    validate_resolution_row,
)


def record_polymarket_real_corpus(
    config: PolymarketRealCorpusRecorderConfig,
) -> PolymarketRealCorpusRecorderResult:
    """Record read-only Polymarket market facts into Phase 2 raw corpus artifacts."""

    run_dir = config.run_dir
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(
                f"recorder run_dir already exists: {run_dir}; "
                "set overwrite_existing=True to replace it"
            )
        shutil.rmtree(run_dir)
    config.raw_dir.mkdir(parents=True)

    provider_failures = _provider_failures(config)
    market_candidates = [] if provider_failures else _discover_market_rows(config)
    book_candidates = _orderbook_rows(market_candidates, config)
    trade_candidates = _trade_rows(market_candidates, config)
    resolution_candidates = _resolution_rows(market_candidates, config)

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
        resolution, resolution_reasons = validate_resolution_row(
            market=market,
            resolution_rows=resolution_candidates,
        )
        reasons = sorted(set(market_reasons + book_reasons + trade_reasons + resolution_reasons))
        if reasons:
            rejected_rows.append(
                {
                    "market_id": market.get("market_id"),
                    "slug": market.get("slug"),
                    "market_family": market.get("market_family"),
                    "reject_reasons": reasons,
                }
            )
            continue
        accepted_markets.append(market)
        raw_payloads["raw_polymarket_markets.jsonl"].append(_raw_market_row(market))
        raw_payloads["raw_polymarket_orderbooks.jsonl"].extend(books)
        raw_payloads["raw_polymarket_trades.jsonl"].extend(trades)
        if resolution is not None:
            raw_payloads["raw_polymarket_resolutions.jsonl"].append(resolution)

    candles, candle_reasons = validate_btc_feature_candles(
        mock_btc_feature_candle_rows(accepted_markets, config)
    )
    if candle_reasons:
        for market in accepted_markets:
            rejected_rows.append(
                {
                    "market_id": market.get("market_id"),
                    "slug": market.get("slug"),
                    "market_family": market.get("market_family"),
                    "reject_reasons": candle_reasons,
                }
            )
        raw_payloads = empty_raw_payloads()
        accepted_markets = []
    else:
        raw_payloads["raw_binance_btcusdt_klines.jsonl"].extend(candles)

    _sort_raw_payloads(raw_payloads)
    _write_raw_files(config.raw_dir, raw_payloads)

    phase2_result = None
    phase2_error: str | None = None
    if config.build_phase2_corpus and raw_payloads["raw_polymarket_markets.jsonl"]:
        try:
            phase2_result = build_polymarket_btc_corpus(
                PolymarketCorpusBuildConfig(
                    input_dir=config.raw_dir,
                    output_dir=config.corpus_dir,
                    created_at=config.created_at,
                    market_families=tuple(config.market_families),  # type: ignore[arg-type]
                    sample_interval_seconds=config.resolved_sampling_policy_seconds(),
                    overwrite_existing=True,
                )
            )
        except Exception as exc:
            phase2_error = str(exc)

    artifact_paths = {
        "real_corpus_recorder_manifest": run_dir / "real_corpus_recorder_manifest.json",
        "real_corpus_recorder_report": run_dir / "real_corpus_recorder_report.json",
        "real_corpus_rejected_rows": run_dir / "real_corpus_rejected_rows.jsonl",
        **{filename: config.raw_dir / filename for filename in RAW_CORPUS_FILENAMES},
    }
    _write_jsonl(artifact_paths["real_corpus_rejected_rows"], rejected_rows)
    report = _recorder_report(
        config=config,
        raw_payloads=raw_payloads,
        rejected_rows=rejected_rows,
        phase2_result=phase2_result,
        phase2_error=phase2_error,
        provider_failures=provider_failures,
    )
    manifest = _recorder_manifest(
        config=config,
        raw_payloads=raw_payloads,
        report=report,
        phase2_result=phase2_result,
    )
    _write_json(artifact_paths["real_corpus_recorder_report"], report)
    _write_json(artifact_paths["real_corpus_recorder_manifest"], manifest)
    if phase2_result is not None:
        artifact_paths.update(
            {f"phase2_{name}": path for name, path in phase2_result.artifact_paths.items()}
        )
    artifact_hashes = {
        name: _sha256_file(path) for name, path in sorted(artifact_paths.items()) if path.exists()
    }
    return PolymarketRealCorpusRecorderResult(
        run_dir=run_dir,
        raw_dir=config.raw_dir,
        corpus_dir=None if phase2_result is None else phase2_result.output_dir,
        artifact_paths=artifact_paths,
        artifact_hashes=artifact_hashes,
        manifest=manifest,
        report=report,
        phase2_result=phase2_result,
    )


def _discover_market_rows(config: PolymarketRealCorpusRecorderConfig) -> list[dict[str, Any]]:
    return discover_mock_market_rows(config)


def _provider_failures(config: PolymarketRealCorpusRecorderConfig) -> list[dict[str, Any]]:
    if config.mock_public_data:
        return []
    return [
        {
            "provider": "polymarket_gamma",
            "provider_stage": "market_discovery",
            "reject_reasons": ["real_public_collection_not_configured"],
            "details": "Gamma market normalization is not wired for production collection yet.",
        },
        {
            "provider": "polymarket_clob",
            "provider_stage": "orderbook_and_trade_collection",
            "reject_reasons": ["real_public_collection_not_configured"],
            "details": "CLOB read-only orderbook/trade collection is not wired yet.",
        },
        {
            "provider": "btc_reference",
            "provider_stage": "feature_candle_collection",
            "reject_reasons": ["real_public_collection_not_configured"],
            "details": "Configured BTC feature candle collection is not wired yet.",
        },
        {
            "provider": "polymarket_resolution",
            "provider_stage": "resolution_collection",
            "reject_reasons": ["real_public_collection_not_configured"],
            "details": "Official settlement reference collection is not wired yet.",
        },
    ]


def _orderbook_rows(
    markets: list[dict[str, Any]],
    config: PolymarketRealCorpusRecorderConfig,
) -> list[dict[str, Any]]:
    return mock_orderbook_rows(markets, config)


def _trade_rows(
    markets: list[dict[str, Any]],
    config: PolymarketRealCorpusRecorderConfig,
) -> list[dict[str, Any]]:
    del config
    return mock_trade_rows(markets)


def _resolution_rows(
    markets: list[dict[str, Any]],
    config: PolymarketRealCorpusRecorderConfig,
) -> list[dict[str, Any]]:
    return mock_resolution_rows(markets, config)


def _validate_market_row(market: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for field_name in (
        "market_id",
        "condition_id",
        "slug",
        "market_family",
        "up_token_id",
        "down_token_id",
        "settlement_rule",
    ):
        if not str(market.get(field_name) or "").strip():
            reasons.append(f"missing_{field_name}")
    if not str(market.get("reference_price_source") or "").strip():
        reasons.append("missing_verified_resolution_source")
    if market.get("up_token_id") == market.get("down_token_id"):
        reasons.append("duplicate_up_down_token_id")
    horizon_ms = int(market.get("horizon_ms") or 0)
    start_ts = int(market.get("market_start_ts") or 0)
    end_ts = int(market.get("market_end_ts") or 0)
    if end_ts <= start_ts:
        reasons.append("invalid_market_window")
    if end_ts - start_ts != horizon_ms:
        reasons.append("market_window_horizon_mismatch")
    for field_name, expected in safety_fields().items():
        if market.get(field_name) is not expected:
            reasons.append(f"unsafe_{field_name}")
    return reasons


def _raw_market_row(market: dict[str, Any]) -> dict[str, Any]:
    return {
        "market_id": market["market_id"],
        "condition_id": market["condition_id"],
        "slug": market["slug"],
        "market_family": market["market_family"],
        "horizon_ms": market["horizon_ms"],
        "market_start_ts": market["market_start_ts"],
        "market_end_ts": market["market_end_ts"],
        "settlement_ts": market["settlement_ts"],
        "up_token_id": market["up_token_id"],
        "down_token_id": market["down_token_id"],
        "reference_price_source": market["reference_price_source"],
        "settlement_rule": market["settlement_rule"],
        "raw_public_payload_sha256": market.get("raw_market_sha256"),
        **safety_fields(),
    }


def _recorder_report(
    *,
    config: PolymarketRealCorpusRecorderConfig,
    raw_payloads: dict[str, list[dict[str, Any]]],
    rejected_rows: list[dict[str, Any]],
    phase2_result: Any,
    phase2_error: str | None,
    provider_failures: list[dict[str, Any]],
) -> dict[str, Any]:
    reject_counts = Counter()
    for row in rejected_rows:
        for reason in row.get("reject_reasons", []):
            reject_counts[reason] += 1
    market_count = len(raw_payloads["raw_polymarket_markets.jsonl"])
    phase2_corpus_manifest_sha256 = None
    if phase2_result is not None:
        phase2_corpus_manifest_sha256 = phase2_result.artifact_hashes.get("corpus_manifest")
    phase2_corpus_build_eligible = (
        phase2_result is not None and market_count > 0 and not phase2_error
    )
    real_historical_training_eligible = phase2_corpus_build_eligible and not config.mock_public_data
    public_collection_status = (
        "mocked"
        if config.mock_public_data
        else "blocked_fail_closed"
        if provider_failures
        else "completed"
    )
    public_collection_reason_codes = sorted(
        {reason for row in provider_failures for reason in row.get("reject_reasons", [])}
    )
    return {
        "schema_version": POLYMARKET_REAL_CORPUS_RECORDER_SCHEMA_VERSION,
        "phase": POLYMARKET_REAL_CORPUS_RECORDER_PHASE,
        "run_id": config.run_id,
        "started_at": config.started_at,
        "ended_at": config.ended_at,
        "wall_clock_duration_seconds": 0.0,
        "market_families": list(config.market_families),
        "sampling_policy": config.resolved_sampling_policy_seconds(),
        "requested_live_public_collection": not config.mock_public_data,
        "public_collection_status": public_collection_status,
        "public_collection_reason_codes": public_collection_reason_codes,
        "live_polymarket_data": real_historical_training_eligible,
        "live_btc_reference_data": real_historical_training_eligible,
        "deterministic_replay": config.mock_public_data,
        "mock_public_data_used": config.mock_public_data,
        "synthetic_public_data_used": config.mock_public_data,
        "synthetic_corpus_used": config.mock_public_data,
        "raw_polymarket_market_count": market_count,
        "raw_orderbook_row_count": len(raw_payloads["raw_polymarket_orderbooks.jsonl"]),
        "raw_trade_row_count": len(raw_payloads["raw_polymarket_trades.jsonl"]),
        "raw_btc_candle_row_count": len(raw_payloads["raw_binance_btcusdt_klines.jsonl"]),
        "raw_resolution_count": len(raw_payloads["raw_polymarket_resolutions.jsonl"]),
        "rejected_row_count": len(rejected_rows),
        "reject_reason_counts": dict(sorted(reject_counts.items())),
        "training_eligible": real_historical_training_eligible,
        "phase2_corpus_build_eligible": phase2_corpus_build_eligible,
        "real_historical_training_eligible": real_historical_training_eligible,
        "manual_live_evidence_eligible": real_historical_training_eligible,
        "phase2_corpus_built": phase2_result is not None,
        "phase2_error": phase2_error,
        "phase2_corpus_manifest_sha256": phase2_corpus_manifest_sha256,
        "real_historical_corpus_used": real_historical_training_eligible,
        "fixture_corpus_used": False,
        **safety_fields(),
    }


def _recorder_manifest(
    *,
    config: PolymarketRealCorpusRecorderConfig,
    raw_payloads: dict[str, list[dict[str, Any]]],
    report: dict[str, Any],
    phase2_result: Any,
) -> dict[str, Any]:
    raw_paths = {filename: config.raw_dir / filename for filename in RAW_CORPUS_FILENAMES}
    return {
        "schema_version": POLYMARKET_REAL_CORPUS_RECORDER_SCHEMA_VERSION,
        "phase": POLYMARKET_REAL_CORPUS_RECORDER_PHASE,
        "run_id": config.run_id,
        "created_at": config.created_at,
        "started_at": config.started_at,
        "ended_at": config.ended_at,
        "config": config.to_dict(),
        "raw_artifact_hashes": {
            filename: _sha256_file(path) for filename, path in raw_paths.items() if path.exists()
        },
        "raw_artifact_row_counts": {
            filename: len(raw_payloads[filename]) for filename in RAW_CORPUS_FILENAMES
        },
        "training_eligible": report["training_eligible"],
        "phase2_corpus_build_eligible": report["phase2_corpus_build_eligible"],
        "real_historical_training_eligible": report["real_historical_training_eligible"],
        "manual_live_evidence_eligible": report["manual_live_evidence_eligible"],
        "phase2_corpus_built": phase2_result is not None,
        "phase2_corpus_dir": None if phase2_result is None else str(phase2_result.output_dir),
        "phase2_corpus_manifest_sha256": report["phase2_corpus_manifest_sha256"],
        "mock_public_data_used": report["mock_public_data_used"],
        "synthetic_public_data_used": report["synthetic_public_data_used"],
        "synthetic_corpus_used": report["synthetic_corpus_used"],
        "real_historical_corpus_used": report["real_historical_corpus_used"],
        "fixture_corpus_used": report["fixture_corpus_used"],
        "requested_live_public_collection": report["requested_live_public_collection"],
        "public_collection_status": report["public_collection_status"],
        "public_collection_reason_codes": report["public_collection_reason_codes"],
        **safety_fields(),
    }


def _sort_raw_payloads(raw_payloads: dict[str, list[dict[str, Any]]]) -> None:
    for filename, rows in raw_payloads.items():
        raw_payloads[filename] = sorted(rows, key=lambda row: json.dumps(row, sort_keys=True))


def _write_raw_files(raw_dir: Path, raw_payloads: dict[str, list[dict[str, Any]]]) -> None:
    for filename in RAW_CORPUS_FILENAMES:
        _write_jsonl(raw_dir / filename, raw_payloads[filename])


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(_json_ready(row), sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
