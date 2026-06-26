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
    orderbook_failure_explanation,
    validate_market_books,
    validate_trade_rows,
)
from bigan.v8.polymarket.recorder.public_provider import PolymarketRealCorpusPublicProvider
from bigan.v8.polymarket.recorder.resolution import (
    mock_resolution_rows,
    validate_resolution_row,
)


def record_polymarket_real_corpus(
    config: PolymarketRealCorpusRecorderConfig,
    *,
    public_provider: PolymarketRealCorpusPublicProvider | None = None,
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

    (
        provider_failures,
        market_candidates,
        book_candidates,
        trade_candidates,
        candle_candidates,
        resolution_candidates,
    ) = _collect_public_rows(config=config, public_provider=public_provider)

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
                _rejected_market(
                    market,
                    reasons,
                    reason_details=_reason_details_for_capture_rejection(
                        market=market,
                        book_rows=book_candidates,
                        config=config,
                        book_reasons=book_reasons,
                    ),
                )
            )
            continue
        accepted_markets.append(market)
        raw_payloads["raw_polymarket_markets.jsonl"].append(_raw_market_row(market))
        raw_payloads["raw_polymarket_orderbooks.jsonl"].extend(books)
        raw_payloads["raw_polymarket_trades.jsonl"].extend(trades)
        if resolution is not None:
            raw_payloads["raw_polymarket_resolutions.jsonl"].append(resolution)

    candle_rows = (
        mock_btc_feature_candle_rows(accepted_markets, config)
        if config.mock_public_data
        else candle_candidates
    )
    candles, candle_reasons = validate_btc_feature_candles(
        candle_rows if accepted_markets else []
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


def _collect_public_rows(
    *,
    config: PolymarketRealCorpusRecorderConfig,
    public_provider: PolymarketRealCorpusPublicProvider | None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    if config.mock_public_data:
        market_candidates = discover_mock_market_rows(config)
        return (
            [],
            market_candidates,
            mock_orderbook_rows(market_candidates, config),
            mock_trade_rows(market_candidates),
            [],
            mock_resolution_rows(market_candidates, config),
        )
    if public_provider is None:
        return (_provider_not_configured_failures(), [], [], [], [], [])

    provider_failures = _provider_safety_failures(public_provider)
    if provider_failures:
        return (provider_failures, [], [], [], [], [])

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
    resolution_candidates = _call_provider_stage(
        provider="polymarket_resolution",
        provider_stage="resolution_collection",
        failures=provider_failures,
        callback=lambda: public_provider.resolution_rows(market_candidates, config),
    )
    return (
        provider_failures,
        market_candidates,
        book_candidates,
        trade_candidates,
        candle_candidates,
        resolution_candidates,
    )


def _provider_not_configured_failures() -> list[dict[str, Any]]:
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


def _provider_safety_failures(provider: PolymarketRealCorpusPublicProvider) -> list[dict[str, Any]]:
    expected = {
        "read_only": True,
        "write_capable": False,
        "paper_only": True,
        "capital_at_risk": False,
        "broker_exchange_write_enabled": False,
        "live_exchange_write_enabled": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }
    bad_fields = [
        field_name
        for field_name, expected_value in expected.items()
        if getattr(provider, field_name, None) is not expected_value
    ]
    if not bad_fields:
        return []
    return [
        {
            "provider": type(provider).__name__,
            "provider_stage": "provider_safety_validation",
            "reject_reasons": ["unsafe_public_provider"],
            "details": "Unsafe public provider flags: " + ", ".join(sorted(bad_fields)),
        }
    ]


def _call_provider_stage(
    *,
    provider: str,
    provider_stage: str,
    failures: list[dict[str, Any]],
    callback: Any,
) -> list[dict[str, Any]]:
    try:
        rows = callback()
    except Exception as exc:
        failures.append(
            {
                "provider": provider,
                "provider_stage": provider_stage,
                "reject_reasons": _exception_reason_codes(exc),
                "details": str(exc),
            }
        )
        return []
    return [dict(row) for row in rows]


def _exception_reason_codes(exc: Exception) -> list[str]:
    reason_codes = getattr(exc, "reason_codes", ())
    if reason_codes:
        return sorted({str(reason) for reason in reason_codes})
    return ["real_public_collection_provider_error"]


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


def _rejected_market(
    market: dict[str, Any],
    reasons: list[str],
    *,
    reason_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "market_id": market.get("market_id"),
        "slug": market.get("slug"),
        "market_family": market.get("market_family"),
        "reject_reasons": sorted(set(reasons)),
    }
    if reason_details:
        row["reason_details"] = reason_details
    return row


def _reason_details_for_capture_rejection(
    *,
    market: dict[str, Any],
    book_rows: list[dict[str, Any]],
    config: PolymarketRealCorpusRecorderConfig,
    book_reasons: list[str],
) -> dict[str, Any] | None:
    orderbook_reasons = {
        "missing_complete_up_down_orderbook",
        "insufficient_decision_timestamps",
        "unknown_token_id",
        "token_id_outcome_mismatch",
        "stale_or_future_orderbook",
        "invalid_orderbook_prices",
    }
    if not (set(book_reasons) & orderbook_reasons):
        return None
    return {
        "orderbook_completeness": orderbook_failure_explanation(
            market=market,
            book_rows=book_rows,
            config=config,
        )
    }


def _raw_market_row(market: dict[str, Any]) -> dict[str, Any]:
    row = {
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
    reference_price_start = market.get("reference_price_start")
    if reference_price_start is None:
        reference_price_start = market.get("reference_price_at_start")
    if reference_price_start is not None:
        row["reference_price_start"] = reference_price_start
        row["reference_price_at_start"] = reference_price_start
    if market.get("reference_price_start_source_type") is not None:
        row["reference_price_start_source_type"] = market[
            "reference_price_start_source_type"
        ]
    return row


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
    live_polymarket_data_read = (
        not config.mock_public_data
        and market_count > 0
        and len(raw_payloads["raw_polymarket_orderbooks.jsonl"]) > 0
    )
    live_btc_reference_data_read = (
        not config.mock_public_data and len(raw_payloads["raw_binance_btcusdt_klines.jsonl"]) > 0
    )
    real_historical_training_eligible = (
        phase2_corpus_build_eligible
        and live_polymarket_data_read
        and live_btc_reference_data_read
        and not provider_failures
    )
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
        "live_polymarket_data_read": live_polymarket_data_read,
        "live_btc_reference_data_read": live_btc_reference_data_read,
        "live_polymarket_data": live_polymarket_data_read,
        "live_btc_reference_data": live_btc_reference_data_read,
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
        "live_polymarket_data_read": report["live_polymarket_data_read"],
        "live_btc_reference_data_read": report["live_btc_reference_data_read"],
        "live_polymarket_data": report["live_polymarket_data"],
        "live_btc_reference_data": report["live_btc_reference_data"],
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
