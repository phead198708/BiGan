"""Round-scoped pending capture and asynchronous settlement finalization."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.corpus import (
    RAW_CORPUS_FILENAMES,
    PolymarketCorpusBuildConfig,
    build_polymarket_btc_corpus,
)
from bigan.v8.polymarket.corpus.contracts import safety_fields
from bigan.v8.polymarket.live import (
    finalize_polymarket_round_artifacts,
    write_polymarket_round_lifecycle_indexes,
)
from bigan.v8.polymarket.live.contracts import (
    BinanceBTCCandle,
    PolymarketLiveMarket,
    PolymarketLiveOrderBook,
    PolymarketLivePaperConfig,
    PolymarketLiveTrade,
)
from bigan.v8.polymarket.recorder.btc_reference import validate_btc_feature_candles
from bigan.v8.polymarket.recorder.chainlink_rtds import (
    CHAINLINK_RTDS_COLLECTION_REPORT_FILENAME,
    CHAINLINK_RTDS_CORPUS_FILENAME,
    CHAINLINK_RTDS_CORPUS_MANIFEST_FILENAME,
    CHAINLINK_RTDS_RAW_FILENAME,
    ChainlinkRTDSSnapshotSource,
)
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
    orderbook_failure_explanation,
    validate_market_books,
    validate_trade_rows,
)
from bigan.v8.polymarket.recorder.public_provider import PolymarketRealCorpusPublicProvider
from bigan.v8.polymarket.recorder.resolution import validate_resolution_row
from bigan.v8.polymarket.rules import build_btc_updown_resolution_rule, resolve_polymarket_rule
from bigan.v8.polymarket.storage import (
    V8_TRAINING_CORPUS_ROOT,
    export_trainable_corpus,
    round_corpus_id_from_corpus_dir,
)

ASYNC_SETTLEMENT_SCHEMA_VERSION = "bigan-v8-polymarket-async-settlement-v2"
PENDING_CAPTURE_PHASE = "polymarket_pending_round_capture"
PENDING_FINALIZATION_PHASE = "polymarket_pending_round_finalization"
PROVIDER_RAW_DIRNAME = "provider_raw"
CHAINLINK_RTDS_LOOKBACK_MS = 120_000


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
    chainlink_rtds_collector: ChainlinkRTDSSnapshotSource | None = None,
    feature_enrichment_max_attempts: int = 40,
) -> PendingRoundCaptureResult:
    """Capture one round's market facts without waiting for delayed settlement."""

    if feature_enrichment_max_attempts <= 0:
        raise ValueError("feature_enrichment_max_attempts must be positive")
    run_dir = config.run_dir
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(
                f"pending capture run_dir already exists: {run_dir}; "
                "set overwrite_existing=True to replace it"
            )
        shutil.rmtree(run_dir)
    config.raw_dir.mkdir(parents=True)
    provider_raw_dir = run_dir / PROVIDER_RAW_DIRNAME
    provider_raw_dir.mkdir(parents=True)

    provider_failures = _provider_safety_failures(public_provider)
    market_candidates: list[dict[str, Any]] = []
    book_candidates: list[dict[str, Any]] = []
    trade_candidates: list[dict[str, Any]] = []
    candle_candidates: list[dict[str, Any]] = []
    chainlink_candidates: list[dict[str, Any]] = []
    chainlink_collection_report = _empty_chainlink_collection_report()
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
        if chainlink_rtds_collector is not None:
            chainlink_candidates = chainlink_rtds_collector.rows()
            chainlink_collection_report = chainlink_rtds_collector.collection_report()

    provider_chainlink_rows, provider_chainlink_reasons = _causal_chainlink_rows_for_markets(
        rows=chainlink_candidates,
        markets=market_candidates,
    )

    provider_raw_payloads = _provider_raw_payloads(
        market_rows=market_candidates,
        orderbook_rows=book_candidates,
        trade_rows=trade_candidates,
        btc_candle_rows=candle_candidates,
    )
    _sort_raw_payloads(provider_raw_payloads)
    _write_raw_files(provider_raw_dir, provider_raw_payloads)
    _write_jsonl(provider_raw_dir / CHAINLINK_RTDS_RAW_FILENAME, provider_chainlink_rows)

    feature_provider_failures = [
        row
        for row in provider_failures
        if row.get("provider_stage") == "feature_candle_collection"
    ]
    fatal_provider_failures = [
        row
        for row in provider_failures
        if row.get("provider_stage") != "feature_candle_collection"
    ]
    raw_payloads = empty_raw_payloads()
    rejected_rows: list[dict[str, Any]] = list(fatal_provider_failures)
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

    candles, candle_reasons = validate_btc_feature_candles(
        candle_candidates if accepted_markets else []
    )
    (
        candles,
        post_market_close_candle_count,
        _,
    ) = _causal_feature_candles_for_markets(
        rows=candles,
        markets=accepted_markets,
    )
    if accepted_markets and not candles:
        candle_reasons = sorted(
            {*candle_reasons, "missing_btc_feature_candles"}
        )
    pending_feature_enrichment = _feature_enrichment_can_retry(
        accepted_markets=accepted_markets,
        candle_reasons=candle_reasons,
        feature_provider_failures=feature_provider_failures,
    )
    feature_enrichment_reason_codes = sorted(
        {
            *candle_reasons,
            *(
                reason
                for row in feature_provider_failures
                for reason in row.get("reject_reasons", [])
            ),
        }
    )
    if candle_reasons and not pending_feature_enrichment:
        for market in accepted_markets:
            rejected_rows.append(_rejected_market(market, candle_reasons))
        raw_payloads = empty_raw_payloads()
        accepted_markets = []
    elif not candle_reasons:
        raw_payloads["raw_binance_btcusdt_klines.jsonl"].extend(candles)

    raw_chainlink_rows, raw_chainlink_reasons = _causal_chainlink_rows_for_markets(
        rows=chainlink_candidates,
        markets=accepted_markets,
    )

    _sort_raw_payloads(raw_payloads)
    _write_raw_files(config.raw_dir, raw_payloads)
    _write_jsonl(config.raw_dir / CHAINLINK_RTDS_RAW_FILENAME, raw_chainlink_rows)
    _write_json(
        run_dir / CHAINLINK_RTDS_COLLECTION_REPORT_FILENAME,
        _round_chainlink_collection_report(
            collection_report=chainlink_collection_report,
            raw_rows=raw_chainlink_rows,
            accepted_markets=accepted_markets,
            reason_codes=sorted(
                set(provider_chainlink_reasons + raw_chainlink_reasons)
            ),
        ),
    )
    artifact_paths = _pending_capture_paths(
        run_dir,
        config.raw_dir,
        provider_raw_dir,
    )
    _write_jsonl(artifact_paths["pending_round_rejected_rows"], rejected_rows)
    report = _pending_capture_report(
        config=config,
        raw_payloads=raw_payloads,
        provider_raw_payloads=provider_raw_payloads,
        rejected_rows=rejected_rows,
        provider_failures=provider_failures,
        raw_chainlink_rows=raw_chainlink_rows,
        provider_chainlink_rows=provider_chainlink_rows,
        chainlink_collection_report=chainlink_collection_report,
        chainlink_reason_codes=sorted(
            set(provider_chainlink_reasons + raw_chainlink_reasons)
        ),
        feature_enrichment_warning_reason_codes=(
            ["feature_enrichment_post_market_close_candle_rejected"]
            if post_market_close_candle_count
            else []
        ),
        feature_enrichment_post_market_close_candle_rejected_count=(
            post_market_close_candle_count
        ),
        pending_feature_enrichment=pending_feature_enrichment,
        feature_enrichment_reason_codes=feature_enrichment_reason_codes,
        feature_enrichment_max_attempts=feature_enrichment_max_attempts,
    )
    manifest = _pending_capture_manifest(
        config=config,
        raw_payloads=raw_payloads,
        provider_raw_payloads=provider_raw_payloads,
        provider_raw_dir=provider_raw_dir,
        report=report,
        raw_chainlink_rows=raw_chainlink_rows,
        provider_chainlink_rows=provider_chainlink_rows,
        feature_enrichment_warning_reason_codes=(
            ["feature_enrichment_post_market_close_candle_rejected"]
            if post_market_close_candle_count
            else []
        ),
        feature_enrichment_post_market_close_candle_rejected_count=(
            post_market_close_candle_count
        ),
        pending_feature_enrichment=pending_feature_enrichment,
        feature_enrichment_reason_codes=feature_enrichment_reason_codes,
        feature_enrichment_max_attempts=feature_enrichment_max_attempts,
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
    provider_raw_dir = resolved_run_dir / PROVIDER_RAW_DIRNAME
    raw_payloads = _read_raw_payloads(raw_dir)
    provider_raw_payloads = _read_raw_payloads(provider_raw_dir)
    raw_chainlink_rows = _read_jsonl(raw_dir / CHAINLINK_RTDS_RAW_FILENAME)
    provider_chainlink_rows = _read_jsonl(
        provider_raw_dir / CHAINLINK_RTDS_RAW_FILENAME
    )
    feature_enrichment_report = _read_json_or_empty(
        resolved_run_dir / "pending_round_feature_enrichment_report.json"
    )
    if capture_manifest.get("pending_feature_enrichment") is True:
        feature_enrichment_report = _attempt_pending_feature_enrichment(
            run_dir=resolved_run_dir,
            config=config,
            public_provider=public_provider,
            raw_payloads=raw_payloads,
            provider_raw_payloads=provider_raw_payloads,
            raw_chainlink_rows=raw_chainlink_rows,
            provider_chainlink_rows=provider_chainlink_rows,
            capture_manifest=capture_manifest,
        )
        capture_manifest = _read_json(manifest_path)
        raw_payloads = _read_raw_payloads(raw_dir)
        provider_raw_payloads = _read_raw_payloads(provider_raw_dir)
        if feature_enrichment_report["feature_enrichment_status"] != "recovered":
            return _pending_feature_enrichment_finalization_result(
                run_dir=resolved_run_dir,
                raw_dir=raw_dir,
                provider_raw_dir=provider_raw_dir,
                config=config,
                raw_payloads=raw_payloads,
                provider_raw_payloads=provider_raw_payloads,
                raw_chainlink_rows=raw_chainlink_rows,
                provider_chainlink_rows=provider_chainlink_rows,
                feature_enrichment_report=feature_enrichment_report,
            )
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
        provider_raw_payloads["raw_polymarket_resolutions.jsonl"] = (
            _preferred_resolution_candidates(
                existing_resolution_candidates=provider_raw_payloads[
                    "raw_polymarket_resolutions.jsonl"
                ],
                provider_resolution_candidates=provider_resolution_candidates,
            )
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
                completed_resolution, completion_reasons = (
                    _complete_resolution_reference_prices(
                        market=market,
                        resolution=resolution,
                        chainlink_rows=raw_chainlink_rows,
                    )
                )
                if completion_reasons:
                    rejected_rows.append(
                        _rejected_market(market, completion_reasons)
                    )
                    continue
                if completed_resolution is not None:
                    resolution_rows.append(completed_resolution)
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
    _sort_raw_payloads(provider_raw_payloads)
    _write_raw_files(provider_raw_dir, provider_raw_payloads)
    round_artifact_evidence = _finalize_pending_round_lifecycle_artifacts(
        config=config,
        run_dir=resolved_run_dir,
        raw_payloads=raw_payloads,
        raw_chainlink_rows=raw_chainlink_rows,
        finalization_reason_codes=tuple(
            sorted({reason for row in rejected_rows for reason in row.get("reject_reasons", [])})
        ),
    )

    corpus_dir = None
    phase2_result = None
    exported_training_corpus_dir = None
    chainlink_corpus_evidence = _empty_chainlink_corpus_evidence()
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
            chainlink_corpus_evidence = _attach_chainlink_evidence_to_corpus(
                corpus_dir=corpus_dir,
                raw_rows=raw_chainlink_rows,
                config=config,
            )
            if raw_chainlink_rows and not chainlink_corpus_evidence.get("attached"):
                raise ValueError(
                    "Chainlink decision-time feature integration failed: "
                    + ", ".join(chainlink_corpus_evidence.get("reason_codes") or [])
                )
            round_slug = round_corpus_id_from_corpus_dir(corpus_dir)
            export_provenance = {
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
                "chainlink_decision_time_evidence": chainlink_corpus_evidence,
                "mock_public_data_used": False,
                "synthetic_public_data_used": False,
                "synthetic_corpus_used": False,
                **safety_fields(),
            }
            exported_training_corpus_dir = _export_or_reuse_matching_corpus(
                corpus_dir=corpus_dir,
                corpus_id=round_slug,
                destination_root=destination_root,
                overwrite_existing=overwrite_existing,
                provenance=export_provenance,
            )
        except Exception as exc:  # noqa: BLE001
            phase2_error = str(exc)

    artifact_paths = _pending_finalization_paths(
        resolved_run_dir,
        raw_dir,
        provider_raw_dir,
    )
    _write_jsonl(artifact_paths["pending_round_finalization_rejected_rows"], rejected_rows)
    report = _pending_finalization_report(
        config=config,
        raw_payloads=raw_payloads,
        provider_raw_payloads=provider_raw_payloads,
        rejected_rows=rejected_rows,
        phase2_result=phase2_result,
        phase2_error=phase2_error,
        exported_training_corpus_dir=exported_training_corpus_dir,
        round_artifact_evidence=round_artifact_evidence,
        raw_chainlink_rows=raw_chainlink_rows,
        provider_chainlink_rows=provider_chainlink_rows,
        chainlink_corpus_evidence=chainlink_corpus_evidence,
        feature_enrichment_report=feature_enrichment_report,
    )
    finalization_manifest = _pending_finalization_manifest(
        config=config,
        raw_payloads=raw_payloads,
        provider_raw_payloads=provider_raw_payloads,
        provider_raw_dir=provider_raw_dir,
        report=report,
        phase2_result=phase2_result,
        exported_training_corpus_dir=exported_training_corpus_dir,
        round_artifact_evidence=round_artifact_evidence,
        raw_chainlink_rows=raw_chainlink_rows,
        provider_chainlink_rows=provider_chainlink_rows,
        chainlink_corpus_evidence=chainlink_corpus_evidence,
        feature_enrichment_report=feature_enrichment_report,
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


def _attempt_pending_feature_enrichment(
    *,
    run_dir: Path,
    config: PolymarketRealCorpusRecorderConfig,
    public_provider: PolymarketRealCorpusPublicProvider,
    raw_payloads: dict[str, list[dict[str, Any]]],
    provider_raw_payloads: dict[str, list[dict[str, Any]]],
    raw_chainlink_rows: list[dict[str, Any]],
    provider_chainlink_rows: list[dict[str, Any]],
    capture_manifest: dict[str, Any],
) -> dict[str, Any]:
    report_path = run_dir / "pending_round_feature_enrichment_report.json"
    manifest_path = (
        run_dir / "pending_round_feature_enrichment_manifest.json"
    )
    capture_report_path = run_dir / "pending_round_capture_report.json"
    capture_manifest_path = run_dir / "pending_round_capture_manifest.json"
    capture_report = _read_json(capture_report_path)
    previous_attempt_count = int(
        capture_manifest.get("feature_enrichment_attempt_count") or 0
    )
    attempt_count = previous_attempt_count + 1
    max_attempts = int(
        capture_manifest.get("feature_enrichment_max_attempts") or 0
    )
    if max_attempts <= 0:
        max_attempts = 1
    market_rows = raw_payloads["raw_polymarket_markets.jsonl"]
    failures: list[dict[str, Any]] = []
    candle_candidates = _call_provider_stage(
        provider="btc_reference",
        provider_stage="feature_candle_enrichment",
        failures=failures,
        callback=lambda: public_provider.btc_feature_candle_rows(
            market_rows,
            config,
        ),
    )
    candles, candle_reasons = validate_btc_feature_candles(
        candle_candidates
    )
    (
        candles,
        post_market_close_candle_count,
        max_market_end_ts,
    ) = _causal_feature_candles_for_markets(
        rows=candles,
        markets=market_rows,
    )
    blocking_reason_codes = sorted(
        {
            *candle_reasons,
            *(
                str(reason)
                for row in failures
                for reason in row.get("reject_reasons", [])
            ),
            *(
                ["missing_btc_feature_candles"]
                if not candles
                else []
            ),
        }
    )
    warning_reason_codes = (
        ["feature_enrichment_post_market_close_candle_rejected"]
        if post_market_close_candle_count
        else []
    )
    recovered = bool(
        market_rows
        and candles
        and not failures
        and not candle_reasons
    )
    exhausted = not recovered and attempt_count >= max_attempts
    status = (
        "recovered"
        if recovered
        else "blocked_fail_closed"
        if exhausted
        else "pending_feature_enrichment"
    )
    source_counts = Counter(str(row.get("source") or "unknown") for row in candles)
    max_available_at_ts = max(
        (
            int(
                row.get("available_at_ts")
                or row.get("close_time")
                or 0
            )
            for row in candles
        ),
        default=0,
    )
    if recovered:
        raw_payloads["raw_binance_btcusdt_klines.jsonl"] = [
            dict(row) for row in candles
        ]
        provider_raw_payloads[
            "raw_binance_btcusdt_klines.jsonl"
        ] = [dict(row) for row in candle_candidates]
        _sort_raw_payloads(raw_payloads)
        _write_raw_files(config.raw_dir, raw_payloads)
        _sort_raw_payloads(provider_raw_payloads)
        _write_raw_files(run_dir / PROVIDER_RAW_DIRNAME, provider_raw_payloads)

    report = {
        "schema_version": ASYNC_SETTLEMENT_SCHEMA_VERSION,
        "phase": "polymarket_pending_round_feature_enrichment",
        "run_id": config.run_id,
        "feature_enrichment_status": status,
        "feature_enrichment_attempt_count": attempt_count,
        "feature_enrichment_max_attempts": max_attempts,
        "feature_enrichment_recovered": recovered,
        "feature_enrichment_exhausted": exhausted,
        "feature_enrichment_reason_codes": (
            [] if recovered else blocking_reason_codes
        ),
        "feature_enrichment_warning_reason_codes": warning_reason_codes,
        "feature_enrichment_source_distribution": dict(
            sorted(source_counts.items())
        ),
        "feature_enrichment_candle_row_count": len(candles),
        "feature_enrichment_provider_candle_row_count": len(
            candle_candidates
        ),
        "feature_enrichment_post_market_close_candle_rejected_count": (
            post_market_close_candle_count
        ),
        "feature_enrichment_candle_max_available_at_ts": (
            max_available_at_ts
        ),
        "feature_enrichment_market_end_ts": max_market_end_ts,
        "feature_enrichment_accepted_causality_violation_count": 0,
        "feature_enrichment_rejected_causality_violation_count": (
            post_market_close_candle_count
        ),
        "feature_enrichment_causality_validation_passed": (
            bool(candles)
            and all(
                int(
                    row.get("available_at_ts")
                    or row.get("close_time")
                    or 0
                )
                <= max_market_end_ts
                for row in candles
            )
        ),
        "resolution_provider_called": False,
        "outcome_or_pnl_fields_accessed": False,
        **safety_fields(),
    }
    _write_json(report_path, report)
    enrichment_manifest = {
        "schema_version": ASYNC_SETTLEMENT_SCHEMA_VERSION,
        "phase": "polymarket_pending_round_feature_enrichment",
        "run_id": config.run_id,
        "report_path": str(report_path),
        "report_sha256": _sha256_file(report_path),
        "raw_artifact_hashes": {
            filename: _sha256_file(config.raw_dir / filename)
            for filename in RAW_CORPUS_FILENAMES
            if (config.raw_dir / filename).exists()
        },
        "provider_raw_artifact_hashes": {
            filename: _sha256_file(
                run_dir / PROVIDER_RAW_DIRNAME / filename
            )
            for filename in RAW_CORPUS_FILENAMES
            if (run_dir / PROVIDER_RAW_DIRNAME / filename).exists()
        },
        "feature_enrichment_status": status,
        "feature_enrichment_attempt_count": attempt_count,
        "feature_enrichment_recovered": recovered,
        "resolution_provider_called": False,
        "outcome_or_pnl_fields_accessed": False,
        **safety_fields(),
    }
    _write_json(manifest_path, enrichment_manifest)

    capture_report.update(
        {
            "pending_feature_enrichment": not recovered
            and not exhausted,
            "pending_resolution": recovered,
            "capture_status": (
                "pending_resolution"
                if recovered
                else "blocked_fail_closed"
                if exhausted
                else "pending_feature_enrichment"
            ),
            "feature_enrichment_attempt_count": attempt_count,
            "feature_enrichment_recovered": recovered,
            "feature_enrichment_reason_codes": (
                [] if recovered else blocking_reason_codes
            ),
            "feature_enrichment_warning_reason_codes": warning_reason_codes,
            "raw_btc_candle_row_count": len(
                raw_payloads[
                    "raw_binance_btcusdt_klines.jsonl"
                ]
            ),
            "public_collection_reason_codes": (
                [] if recovered else blocking_reason_codes
            ),
        }
    )
    _write_json(capture_report_path, capture_report)
    capture_manifest.update(
        {
            "pending_feature_enrichment": not recovered
            and not exhausted,
            "pending_resolution": recovered,
            "capture_status": capture_report["capture_status"],
            "feature_enrichment_attempt_count": attempt_count,
            "feature_enrichment_reason_codes": (
                [] if recovered else blocking_reason_codes
            ),
            "feature_enrichment_warning_reason_codes": warning_reason_codes,
            "feature_enrichment_recovered": recovered,
            "training_raw_is_validated_sampled_view": recovered,
            "raw_artifact_hashes": {
                filename: _sha256_file(config.raw_dir / filename)
                for filename in RAW_CORPUS_FILENAMES
                if (config.raw_dir / filename).exists()
            },
            "raw_artifact_row_counts": {
                filename: len(raw_payloads[filename])
                for filename in RAW_CORPUS_FILENAMES
            },
            "provider_raw_artifact_hashes": {
                filename: _sha256_file(
                    run_dir / PROVIDER_RAW_DIRNAME / filename
                )
                for filename in RAW_CORPUS_FILENAMES
                if (
                    run_dir / PROVIDER_RAW_DIRNAME / filename
                ).exists()
            },
            "provider_raw_artifact_row_counts": {
                filename: len(provider_raw_payloads[filename])
                for filename in RAW_CORPUS_FILENAMES
            },
            "feature_enrichment_report_sha256": _sha256_file(
                report_path
            ),
            "feature_enrichment_manifest_sha256": _sha256_file(
                manifest_path
            ),
        }
    )
    _write_json(capture_manifest_path, capture_manifest)
    return report


def _pending_feature_enrichment_finalization_result(
    *,
    run_dir: Path,
    raw_dir: Path,
    provider_raw_dir: Path,
    config: PolymarketRealCorpusRecorderConfig,
    raw_payloads: dict[str, list[dict[str, Any]]],
    provider_raw_payloads: dict[str, list[dict[str, Any]]],
    raw_chainlink_rows: list[dict[str, Any]],
    provider_chainlink_rows: list[dict[str, Any]],
    feature_enrichment_report: dict[str, Any],
) -> PendingRoundFinalizationResult:
    status = str(
        feature_enrichment_report["feature_enrichment_status"]
    )
    pending = status == "pending_feature_enrichment"
    reason_codes = list(
        feature_enrichment_report.get(
            "feature_enrichment_reason_codes"
        )
        or []
    )
    artifact_paths = _pending_finalization_paths(
        run_dir,
        raw_dir,
        provider_raw_dir,
    )
    report = {
        "schema_version": ASYNC_SETTLEMENT_SCHEMA_VERSION,
        "phase": PENDING_FINALIZATION_PHASE,
        "run_id": config.run_id,
        "finalization_status": status,
        "pending_feature_enrichment": pending,
        "pending_resolution": False,
        "feature_enrichment_recovered": False,
        "feature_enrichment_attempt_count": int(
            feature_enrichment_report.get(
                "feature_enrichment_attempt_count"
            )
            or 0
        ),
        "feature_enrichment_reason_codes": reason_codes,
        "feature_enrichment_warning_reason_codes": list(
            feature_enrichment_report.get(
                "feature_enrichment_warning_reason_codes"
            )
            or []
        ),
        "feature_enrichment_post_market_close_candle_rejected_count": int(
            feature_enrichment_report.get(
                "feature_enrichment_post_market_close_candle_rejected_count"
            )
            or 0
        ),
        "resolution_provider_called": False,
        "raw_polymarket_market_count": len(
            raw_payloads["raw_polymarket_markets.jsonl"]
        ),
        "raw_orderbook_row_count": len(
            raw_payloads["raw_polymarket_orderbooks.jsonl"]
        ),
        "raw_trade_row_count": len(
            raw_payloads["raw_polymarket_trades.jsonl"]
        ),
        "raw_btc_candle_row_count": len(
            raw_payloads["raw_binance_btcusdt_klines.jsonl"]
        ),
        "raw_chainlink_price_row_count": len(raw_chainlink_rows),
        "provider_raw_chainlink_price_row_count": len(
            provider_chainlink_rows
        ),
        "raw_resolution_count": 0,
        "reject_reason_counts": dict.fromkeys(
            sorted(set(reason_codes)),
            1,
        ),
        "training_eligible": False,
        "phase2_corpus_built": False,
        "phase2_error": None,
        "exported_training_corpus_dir": None,
        "outcome_or_pnl_fields_accessed": False,
        **safety_fields(),
    }
    _write_json(
        artifact_paths["pending_round_finalization_report"],
        report,
    )
    manifest = {
        "schema_version": ASYNC_SETTLEMENT_SCHEMA_VERSION,
        "phase": PENDING_FINALIZATION_PHASE,
        "run_id": config.run_id,
        "report_path": str(
            artifact_paths["pending_round_finalization_report"]
        ),
        "report_sha256": _sha256_file(
            artifact_paths["pending_round_finalization_report"]
        ),
        "feature_enrichment_report_path": str(
            artifact_paths[
                "pending_round_feature_enrichment_report"
            ]
        ),
        "feature_enrichment_report_sha256": _sha256_file(
            artifact_paths[
                "pending_round_feature_enrichment_report"
            ]
        ),
        "finalization_status": status,
        "pending_feature_enrichment": pending,
        "pending_resolution": False,
        "resolution_provider_called": False,
        "training_eligible": False,
        "exported_training_corpus_dir": None,
        "outcome_or_pnl_fields_accessed": False,
        **safety_fields(),
    }
    _write_json(
        artifact_paths["pending_round_finalization_manifest"],
        manifest,
    )
    artifact_hashes = {
        name: _sha256_file(path)
        for name, path in sorted(artifact_paths.items())
        if path.exists()
    }
    return PendingRoundFinalizationResult(
        run_dir=run_dir,
        raw_dir=raw_dir,
        corpus_dir=None,
        exported_training_corpus_dir=None,
        artifact_paths=artifact_paths,
        artifact_hashes=artifact_hashes,
        manifest=manifest,
        report=report,
    )


def _pending_capture_paths(
    run_dir: Path,
    raw_dir: Path,
    provider_raw_dir: Path,
) -> dict[str, Path]:
    return {
        "pending_round_capture_manifest": run_dir / "pending_round_capture_manifest.json",
        "pending_round_capture_report": run_dir / "pending_round_capture_report.json",
        "pending_round_rejected_rows": run_dir / "pending_round_rejected_rows.jsonl",
        "pending_round_feature_enrichment_report": (
            run_dir / "pending_round_feature_enrichment_report.json"
        ),
        "pending_round_feature_enrichment_manifest": (
            run_dir / "pending_round_feature_enrichment_manifest.json"
        ),
        "raw_polymarket_chainlink_prices": raw_dir / CHAINLINK_RTDS_RAW_FILENAME,
        "polymarket_chainlink_rtds_collection_report": (
            run_dir / CHAINLINK_RTDS_COLLECTION_REPORT_FILENAME
        ),
        **{filename: raw_dir / filename for filename in RAW_CORPUS_FILENAMES},
        **_provider_raw_artifact_paths(provider_raw_dir),
    }


def _pending_finalization_paths(
    run_dir: Path,
    raw_dir: Path,
    provider_raw_dir: Path,
) -> dict[str, Path]:
    return {
        "pending_round_finalization_manifest": (
            run_dir / "pending_round_finalization_manifest.json"
        ),
        "pending_round_finalization_report": run_dir / "pending_round_finalization_report.json",
        "pending_round_finalization_rejected_rows": (
            run_dir / "pending_round_finalization_rejected_rows.jsonl"
        ),
        "pending_round_feature_enrichment_report": (
            run_dir / "pending_round_feature_enrichment_report.json"
        ),
        "pending_round_feature_enrichment_manifest": (
            run_dir / "pending_round_feature_enrichment_manifest.json"
        ),
        "raw_polymarket_chainlink_prices": raw_dir / CHAINLINK_RTDS_RAW_FILENAME,
        "polymarket_chainlink_rtds_collection_report": (
            run_dir / CHAINLINK_RTDS_COLLECTION_REPORT_FILENAME
        ),
        **_pending_round_lifecycle_paths(run_dir),
        **{filename: raw_dir / filename for filename in RAW_CORPUS_FILENAMES},
        **_provider_raw_artifact_paths(provider_raw_dir),
    }


def _provider_raw_artifact_paths(provider_raw_dir: Path) -> dict[str, Path]:
    return {
        f"provider_{filename.removesuffix('.jsonl')}": provider_raw_dir / filename
        for filename in RAW_CORPUS_FILENAMES
    } | {
        "provider_raw_polymarket_chainlink_prices": (
            provider_raw_dir / CHAINLINK_RTDS_RAW_FILENAME
        )
    }


def _pending_round_lifecycle_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "rounds_index": run_dir / "rounds_index.jsonl",
        "training_raw_index": run_dir / "training_raw_index.jsonl",
        "paper_audit_index": run_dir / "paper_audit_index.jsonl",
        "paper_run_summary_latest": run_dir / "paper_run_summary_latest.json",
    }


def _finalize_pending_round_lifecycle_artifacts(
    *,
    config: PolymarketRealCorpusRecorderConfig,
    run_dir: Path,
    raw_payloads: dict[str, list[dict[str, Any]]],
    raw_chainlink_rows: list[dict[str, Any]],
    finalization_reason_codes: tuple[str, ...],
) -> dict[str, Any]:
    markets = raw_payloads["raw_polymarket_markets.jsonl"]
    resolutions_by_market = {
        str(row["market_id"]): row
        for row in raw_payloads["raw_polymarket_resolutions.jsonl"]
        if row.get("market_id")
    }
    if not markets or not resolutions_by_market:
        return _empty_round_artifact_evidence()

    artifact_paths = _pending_round_lifecycle_paths(run_dir)
    round_index_rows = _read_jsonl(artifact_paths["rounds_index"])
    training_index_rows = _read_jsonl(artifact_paths["training_raw_index"])
    paper_audit_index_rows = _read_jsonl(artifact_paths["paper_audit_index"])
    finalized_market_ids = {str(row.get("market_id") or "") for row in round_index_rows}
    round_summaries = _read_existing_round_summaries(
        run_dir=run_dir,
        round_index_rows=round_index_rows,
    )
    latest_run_summary = _read_json_or_empty(artifact_paths["paper_run_summary_latest"])
    live_config = _live_config_for_pending_round(config)
    model_manifest = _pending_round_model_manifest(config)
    model_manifest_sha256 = canonical_json_sha256(model_manifest)
    status = "completed" if not finalization_reason_codes else "blocked_fail_closed"
    recommendation = "continue_paper_run" if status == "completed" else "blocked_fail_closed"

    finalized_count = 0
    for market_row in sorted(markets, key=lambda row: (int(row["market_start_ts"]), str(row["market_id"]))):
        market_id = str(market_row["market_id"])
        if market_id in finalized_market_ids:
            continue
        resolution = resolutions_by_market.get(market_id)
        if resolution is None:
            continue
        live_market = _live_market_from_pending_raw(market_row, resolution)
        settlement_candle = _settlement_candle_from_pending_raw(market_row, resolution)
        finalized = finalize_polymarket_round_artifacts(
            config=live_config,
            run_dir=run_dir,
            rounds_root=run_dir / "rounds",
            market=live_market,
            orderbooks=_live_orderbooks_from_pending_raw(market_id, raw_payloads),
            trades=_live_trades_from_pending_raw(market_id, raw_payloads),
            candle=settlement_candle,
            predictions=[],
            decisions=[],
            ledger_events=[],
            settlement=_settlement_event_from_pending_raw(
                market=market_row,
                resolution=resolution,
                candle=settlement_candle,
            ),
            observability_report=_pending_round_observability_report(
                config=config,
                reason_codes=finalization_reason_codes,
            ),
            completed_round_summaries=round_summaries,
            model_manifest=model_manifest,
            model_manifest_sha256=model_manifest_sha256,
            run_reason_codes=finalization_reason_codes,
            status=status,
            recommendation=recommendation,
            chainlink_prices=_chainlink_rows_for_market(
                rows=raw_chainlink_rows,
                market=market_row,
            ),
        )
        round_summaries.append(finalized["round_summary"])
        index_row = finalized["index_row"]
        round_index_rows.append(index_row)
        paper_audit_index_rows.append(index_row)
        if finalized["training_eligible"]:
            training_index_rows.append(index_row)
        latest_run_summary = finalized["latest_run_summary"]
        write_polymarket_round_lifecycle_indexes(
            artifact_paths=artifact_paths,
            round_index_rows=round_index_rows,
            training_index_rows=training_index_rows,
            paper_audit_index_rows=paper_audit_index_rows,
            latest_run_summary=latest_run_summary,
        )
        finalized_market_ids.add(market_id)
        finalized_count += 1

    if not artifact_paths["paper_run_summary_latest"].exists():
        write_polymarket_round_lifecycle_indexes(
            artifact_paths=artifact_paths,
            round_index_rows=round_index_rows,
            training_index_rows=training_index_rows,
            paper_audit_index_rows=paper_audit_index_rows,
            latest_run_summary=latest_run_summary,
        )
    return {
        "round_artifact_export_mode": "round_finalization_lifecycle",
        "round_artifacts_written": len(round_index_rows),
        "round_artifacts_newly_finalized": finalized_count,
        "training_raw_round_count": len(training_index_rows),
        "paper_audit_round_count": len(paper_audit_index_rows),
        "round_lifecycle_index_paths": {
            name: str(path) for name, path in sorted(artifact_paths.items())
        },
        "latest_run_summary_sha256": (
            None
            if not artifact_paths["paper_run_summary_latest"].exists()
            else _sha256_file(artifact_paths["paper_run_summary_latest"])
        ),
    }


def _empty_round_artifact_evidence() -> dict[str, Any]:
    return {
        "round_artifact_export_mode": "round_finalization_lifecycle",
        "round_artifacts_written": 0,
        "round_artifacts_newly_finalized": 0,
        "training_raw_round_count": 0,
        "paper_audit_round_count": 0,
        "round_lifecycle_index_paths": {},
        "latest_run_summary_sha256": None,
    }


def _live_config_for_pending_round(
    config: PolymarketRealCorpusRecorderConfig,
) -> PolymarketLivePaperConfig:
    return PolymarketLivePaperConfig(
        run_id=config.run_id,
        output_dir=config.output_dir,
        mock_live=config.mock_public_data,
        market_families=tuple(config.market_families),
        created_at=config.created_at,
        started_at=config.started_at,
        overwrite_existing=True,
    )


def _pending_round_model_manifest(config: PolymarketRealCorpusRecorderConfig) -> dict[str, Any]:
    return {
        "run_id": config.run_id,
        "model_version": "pending-round-finalize-only",
        "recorder_run_id": config.run_id,
        "real_historical_corpus_used": not config.mock_public_data,
        "fixture_corpus_used": False,
        "synthetic_corpus_used": config.mock_public_data,
        "fixture_model_used": False,
        "round_finalization_only": True,
        "model_signal_used": False,
        "paper_decision_used": False,
        "paper_audit_only": True,
        **safety_fields(),
    }


def _live_market_from_pending_raw(
    market: dict[str, Any],
    resolution: dict[str, Any],
) -> PolymarketLiveMarket:
    return PolymarketLiveMarket(
        market_id=str(market["market_id"]),
        condition_id=str(market["condition_id"]),
        slug=str(market["slug"]),
        market_family=str(market["market_family"]),
        horizon_ms=int(market["horizon_ms"]),
        market_start_ts=int(market["market_start_ts"]),
        market_end_ts=int(market["market_end_ts"]),
        settlement_ts=int(market["settlement_ts"]),
        up_token_id=str(market["up_token_id"]),
        down_token_id=str(market["down_token_id"]),
        reference_price_source=str(market["reference_price_source"]),
        settlement_rule=str(market["settlement_rule"]),
        reference_price_at_start=float(resolution["reference_price_start"]),
        status="resolved",
        resolution_available=True,
        raw_market_sha256=str(market.get("raw_public_payload_sha256") or ""),
    )


def _live_orderbooks_from_pending_raw(
    market_id: str,
    raw_payloads: dict[str, list[dict[str, Any]]],
) -> list[PolymarketLiveOrderBook]:
    rows = []
    for row in raw_payloads["raw_polymarket_orderbooks.jsonl"]:
        if row.get("market_id") != market_id:
            continue
        rows.append(
            PolymarketLiveOrderBook(
                market_id=market_id,
                token_id=str(row["token_id"]),
                outcome=str(row["outcome"]).upper(),  # type: ignore[arg-type]
                ts=int(row["ts"]),
                received_ts=int(row.get("available_at_ts") or row["ts"]),
                bid_price=float(row["bid_price"]),
                ask_price=float(row["ask_price"]),
                mid_price=float(row["mid_price"]),
                bid_size=float(row.get("bid_size") or 0.0),
                ask_size=float(row.get("ask_size") or 0.0),
                liquidity_depth=float(row.get("liquidity_depth") or 0.0),
                source=str(row.get("source") or "polymarket_corpus_recorder"),
            )
        )
    return rows


def _live_trades_from_pending_raw(
    market_id: str,
    raw_payloads: dict[str, list[dict[str, Any]]],
) -> list[PolymarketLiveTrade]:
    rows = []
    for row in raw_payloads["raw_polymarket_trades.jsonl"]:
        if row.get("market_id") != market_id:
            continue
        rows.append(
            PolymarketLiveTrade(
                market_id=market_id,
                token_id=str(row["token_id"]),
                outcome=str(row["outcome"]).upper(),  # type: ignore[arg-type]
                ts=int(row["ts"]),
                price=float(row["price"]),
                size=float(row.get("size") or 0.0),
                side=str(row.get("side") or "UNKNOWN"),
                source=str(row.get("source") or "polymarket_corpus_recorder"),
            )
        )
    return rows


def _settlement_candle_from_pending_raw(
    market: dict[str, Any],
    resolution: dict[str, Any],
) -> BinanceBTCCandle:
    start = float(resolution["reference_price_start"])
    end = float(resolution["reference_price_end"])
    return BinanceBTCCandle(
        market_id=str(market["market_id"]),
        market_family=str(market["market_family"]),
        open_ts=int(market["market_start_ts"]),
        close_ts=int(market["market_end_ts"]),
        open_price=start,
        close_price=end,
        high_price=max(start, end),
        low_price=min(start, end),
        source=str(resolution.get("reference_price_source") or market["reference_price_source"]),
    )


def _settlement_event_from_pending_raw(
    *,
    market: dict[str, Any],
    resolution: dict[str, Any],
    candle: BinanceBTCCandle,
) -> dict[str, Any]:
    resolution_status = str(resolution["resolution_status"])
    rule = build_btc_updown_resolution_rule(
        market_id=str(market["market_id"]),
        condition_id=str(market["condition_id"]),
        slug=str(market["slug"]),
        market_family=str(market["market_family"]),
        resolution_source=str(resolution.get("reference_price_source") or market["reference_price_source"]),
        candle_open_ts=int(market["market_start_ts"]),
        candle_close_ts=int(market["market_end_ts"]),
        raw_rule_text=str(market["settlement_rule"]),
        unknown_50_50_enabled=True if resolution_status == "unknown_50_50" else None,
    )
    resolved = resolve_polymarket_rule(
        rule,
        reference_price_start=float(resolution["reference_price_start"]),
        reference_price_end=candle.close_price,
        resolution_status=resolution_status,
    )
    return {
        "market_id": str(market["market_id"]),
        "condition_id": str(market["condition_id"]),
        "slug": str(market["slug"]),
        "resolution_status": resolution_status,
        "resolved_outcome": resolved.resolved_outcome,
        "payout_up": resolved.payout_up,
        "payout_down": resolved.payout_down,
        "reference_price_start": resolved.reference_price_start,
        "reference_price_end": resolved.reference_price_end,
        "qty_up_settled": 0.0,
        "qty_down_settled": 0.0,
        "settlement_cashflow": 0.0,
        "settlement_pnl": 0.0,
        "raw_resolution_sha256": resolved.raw_resolution_sha256,
        **safety_fields(),
    }


def _pending_round_observability_report(
    *,
    config: PolymarketRealCorpusRecorderConfig,
    reason_codes: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "schema_version": ASYNC_SETTLEMENT_SCHEMA_VERSION,
        "phase": PENDING_FINALIZATION_PHASE,
        "run_id": config.run_id,
        "operator_status": "completed" if not reason_codes else "blocked_fail_closed",
        "operator_recommendation": (
            "continue_paper_run" if not reason_codes else "blocked_fail_closed"
        ),
        "critical_alert_count": len(set(reason_codes)),
        "critical_reason_codes": sorted(set(reason_codes)),
        "round_finalization_only": True,
        "model_signal_used": False,
        "paper_decision_used": False,
        "paper_audit_only": True,
        **safety_fields(),
    }


def _read_existing_round_summaries(
    *,
    run_dir: Path,
    round_index_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summaries = []
    for row in round_index_rows:
        round_dir = row.get("round_dir")
        if not round_dir:
            continue
        path = run_dir / str(round_dir) / "round_summary.json"
        if path.exists():
            summaries.append(_read_json(path))
    return summaries


def _read_json_or_empty(path: Path) -> dict[str, Any]:
    return _read_json(path) if path.exists() else {}


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


def _complete_resolution_reference_prices(
    *,
    market: dict[str, Any],
    resolution: dict[str, Any],
    chainlink_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[str]]:
    start = _positive_float(resolution.get("reference_price_start"))
    end = _positive_float(resolution.get("reference_price_end"))
    if start is not None and end is not None:
        return dict(resolution), []

    market_start_ts = int(market["market_start_ts"])
    market_end_ts = int(market["market_end_ts"])
    settlement_ts = int(market.get("settlement_ts") or market_end_ts)
    start_row = _latest_chainlink_row_at_or_before(
        rows=chainlink_rows,
        source_ts=market_start_ts,
        available_at_ts=settlement_ts,
    )
    end_row = _latest_chainlink_row_at_or_before(
        rows=chainlink_rows,
        source_ts=market_end_ts,
        available_at_ts=settlement_ts,
    )
    reasons: list[str] = []
    if start is None and start_row is None:
        reasons.append("pending_reference_price_start")
    if end is None and end_row is None:
        reasons.append("pending_reference_price_end")
    if reasons:
        return None, reasons

    completed = dict(resolution)
    if start is None and start_row is not None:
        completed["reference_price_start"] = float(start_row["price"])
        completed["reference_price_start_source_type"] = (
            "polymarket_rtds_chainlink_market_start"
        )
        completed["reference_price_start_source_ts"] = int(start_row["source_ts"])
        completed["reference_price_start_available_at_ts"] = int(
            start_row["available_at_ts"]
        )
    if end is None and end_row is not None:
        completed["reference_price_end"] = float(end_row["price"])
        completed["reference_price_end_source_type"] = (
            "polymarket_rtds_chainlink_market_end"
        )
        completed["reference_price_end_source_ts"] = int(end_row["source_ts"])
        completed["reference_price_end_available_at_ts"] = int(
            end_row["available_at_ts"]
        )
    completed["reference_price_pair_completed_from_chainlink_rtds"] = True
    completed["reference_price_pair_completion_max_input_ts"] = max(
        int(completed.get("reference_price_start_available_at_ts") or 0),
        int(completed.get("reference_price_end_available_at_ts") or 0),
    )
    expected_outcome = (
        "UP"
        if float(completed["reference_price_end"])
        >= float(completed["reference_price_start"])
        else "DOWN"
    )
    resolved_outcome = str(completed.get("resolved_outcome") or "").upper()
    if resolved_outcome in {"UP", "DOWN"} and resolved_outcome != expected_outcome:
        return None, ["chainlink_reference_direction_mismatch_official_outcome"]
    return completed, []


def _latest_chainlink_row_at_or_before(
    *,
    rows: list[dict[str, Any]],
    source_ts: int,
    available_at_ts: int,
) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if int(row.get("source_ts") or 0) <= source_ts
        and int(row.get("available_at_ts") or 0) <= available_at_ts
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            int(row["source_ts"]),
            int(row["available_at_ts"]),
        ),
    )


def _causal_chainlink_rows_for_markets(
    *,
    rows: list[dict[str, Any]],
    markets: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    if not markets:
        return [], ["chainlink_rtds_no_market_window"] if rows else []
    window_start = min(int(row["market_start_ts"]) for row in markets)
    window_end = max(int(row["market_end_ts"]) for row in markets)
    availability_window_end = max(
        int(row.get("settlement_ts") or row["market_end_ts"])
        for row in markets
    )
    accepted: dict[tuple[int, float], dict[str, Any]] = {}
    reason_codes: set[str] = set()
    for row in rows:
        source_ts = _positive_int(row.get("source_ts"))
        available_at_ts = _positive_int(row.get("available_at_ts"))
        price = _positive_float(row.get("price"))
        if source_ts is None or available_at_ts is None or price is None:
            reason_codes.add("chainlink_rtds_invalid_raw_row")
            continue
        if source_ts > available_at_ts:
            reason_codes.add("chainlink_rtds_timestamp_causality_violation")
            continue
        if source_ts < window_start - CHAINLINK_RTDS_LOOKBACK_MS:
            continue
        if source_ts > window_end:
            continue
        if available_at_ts > availability_window_end:
            reason_codes.add("chainlink_rtds_post_settlement_window_excluded")
            continue
        if row.get("source_type") != "polymarket_rtds_chainlink":
            reason_codes.add("chainlink_rtds_source_type_invalid")
            continue
        if any(
            row.get(field_name) is not expected
            for field_name, expected in safety_fields().items()
        ):
            reason_codes.add("chainlink_rtds_safety_contract_invalid")
            continue
        accepted[(source_ts, price)] = dict(row)
    normalized = sorted(
        accepted.values(),
        key=lambda row: (
            int(row["source_ts"]),
            int(row["available_at_ts"]),
            float(row["price"]),
        ),
    )
    if not normalized:
        reason_codes.add("chainlink_rtds_round_rows_unavailable")
    for market in markets:
        market_start_ts = int(market["market_start_ts"])
        if not any(int(row["source_ts"]) <= market_start_ts for row in normalized):
            reason_codes.add("chainlink_rtds_market_start_reference_unavailable")
    return normalized, sorted(reason_codes)


def _chainlink_rows_for_market(
    *,
    rows: list[dict[str, Any]],
    market: dict[str, Any],
) -> list[dict[str, Any]]:
    selected, _ = _causal_chainlink_rows_for_markets(rows=rows, markets=[market])
    return selected


def _round_chainlink_collection_report(
    *,
    collection_report: dict[str, Any],
    raw_rows: list[dict[str, Any]],
    accepted_markets: list[dict[str, Any]],
    reason_codes: list[str],
) -> dict[str, Any]:
    covered_market_count = sum(
        1
        for market in accepted_markets
        if any(
            int(row["source_ts"]) <= int(market["market_start_ts"])
            for row in raw_rows
        )
    )
    return {
        **collection_report,
        "report_scope": "round_causal_window",
        "accepted_market_count": len(accepted_markets),
        "round_raw_price_row_count": len(raw_rows),
        "market_start_reference_covered_market_count": covered_market_count,
        "market_start_reference_missing_market_count": (
            len(accepted_markets) - covered_market_count
        ),
        "round_reason_codes": reason_codes,
        "timestamp_causality_violation_count": sum(
            1
            for row in raw_rows
            if int(row["source_ts"]) > int(row["available_at_ts"])
        ),
        "read_only": True,
        **safety_fields(),
    }


def _empty_chainlink_collection_report() -> dict[str, Any]:
    return {
        "report_type": "polymarket_chainlink_rtds_collection",
        "source_type": "polymarket_rtds_chainlink",
        "raw_price_row_count": 0,
        "decision_critical": False,
        "fail_closed_when_feature_unavailable": True,
        "read_only": True,
        **safety_fields(),
    }


def _attach_chainlink_evidence_to_corpus(
    *,
    corpus_dir: Path,
    raw_rows: list[dict[str, Any]],
    config: PolymarketRealCorpusRecorderConfig,
) -> dict[str, Any]:
    if not raw_rows:
        return _empty_chainlink_corpus_evidence()
    evidence_path = corpus_dir / CHAINLINK_RTDS_CORPUS_FILENAME
    manifest_path = corpus_dir / CHAINLINK_RTDS_CORPUS_MANIFEST_FILENAME
    if not evidence_path.is_file() or not manifest_path.is_file():
        return {
            **_empty_chainlink_corpus_evidence(),
            "reason_codes": ["chainlink_feature_builder_artifacts_missing"],
        }
    manifest = _read_json(manifest_path)
    evidence_sha256 = _sha256_file(evidence_path)
    reasons = []
    if manifest.get("source_type") != "polymarket_rtds_chainlink":
        reasons.append("chainlink_feature_source_type_invalid")
    if manifest.get("decision_time_only") is not True:
        reasons.append("chainlink_feature_decision_time_contract_failed")
    if int(manifest.get("row_count") or 0) != len(raw_rows):
        reasons.append("chainlink_feature_row_count_mismatch")
    if manifest.get("evidence_sha256") != evidence_sha256:
        reasons.append("chainlink_feature_evidence_sha256_mismatch")
    if manifest.get("feature_builder_integration_passed") is not True:
        reasons.append("chainlink_feature_builder_integration_failed")
    if manifest.get("feature_builder_integration_required") is not False:
        reasons.append("chainlink_feature_builder_integration_still_required")
    if int(manifest.get("timestamp_causality_violation_count") or 0) != 0:
        reasons.append("chainlink_feature_timestamp_causality_violation")
    return {
        "attached": not reasons,
        "source_run_id": config.run_id,
        "row_count": len(raw_rows),
        "evidence_filename": CHAINLINK_RTDS_CORPUS_FILENAME,
        "evidence_sha256": evidence_sha256,
        "manifest_filename": CHAINLINK_RTDS_CORPUS_MANIFEST_FILENAME,
        "manifest_sha256": _sha256_file(manifest_path),
        "feature_builder_integration_passed": not reasons,
        "feature_builder_integration_required": bool(reasons),
        "integrated_feature_row_count": int(
            manifest.get("integrated_feature_row_count") or 0
        ),
        "missing_or_invalid_feature_row_count": int(
            manifest.get("missing_or_invalid_feature_row_count") or 0
        ),
        "reason_codes": sorted(set(reasons)),
    }


def _export_or_reuse_matching_corpus(
    *,
    corpus_dir: Path,
    corpus_id: str,
    destination_root: Path | str,
    overwrite_existing: bool,
    provenance: dict[str, Any],
) -> Path:
    root = Path(destination_root).expanduser().resolve()
    target = root / "polymarket" / corpus_id
    if target.exists() and not overwrite_existing:
        provenance_path = target / "training_corpus_provenance.json"
        corpus_manifest_path = target / "polymarket_corpus_manifest.json"
        if provenance_path.exists() and corpus_manifest_path.exists():
            existing = _read_json(provenance_path)
            expected_chainlink = provenance.get("chainlink_decision_time_evidence") or {}
            existing_chainlink = existing.get("chainlink_decision_time_evidence") or {}
            lineage_matches = (
                existing.get("source") == provenance.get("source")
                and existing.get("run_id") == provenance.get("run_id")
                and existing.get("phase2_corpus_manifest_sha256")
                == provenance.get("phase2_corpus_manifest_sha256")
                and existing_chainlink.get("evidence_sha256")
                == expected_chainlink.get("evidence_sha256")
                and _sha256_file(corpus_manifest_path)
                == provenance.get("phase2_corpus_manifest_sha256")
            )
            chainlink_filename = expected_chainlink.get("evidence_filename")
            if chainlink_filename:
                chainlink_path = target / str(chainlink_filename)
                lineage_matches = (
                    lineage_matches
                    and chainlink_path.exists()
                    and _sha256_file(chainlink_path)
                    == expected_chainlink.get("evidence_sha256")
                )
            if lineage_matches:
                return target
        raise FileExistsError(
            "existing training corpus does not match this finalized round lineage: "
            f"{target}"
        )
    return export_trainable_corpus(
        corpus_dir=corpus_dir,
        corpus_id=corpus_id,
        provenance=provenance,
        destination_root=root,
        overwrite_existing=overwrite_existing,
    )


def _empty_chainlink_corpus_evidence() -> dict[str, Any]:
    return {
        "attached": False,
        "row_count": 0,
        "evidence_filename": None,
        "evidence_sha256": None,
        "manifest_filename": None,
        "manifest_sha256": None,
        "feature_builder_integration_passed": False,
        "feature_builder_integration_required": True,
        "integrated_feature_row_count": 0,
        "missing_or_invalid_feature_row_count": 0,
        "reason_codes": ["chainlink_decision_time_evidence_unavailable"],
    }


def _pending_capture_report(
    *,
    config: PolymarketRealCorpusRecorderConfig,
    raw_payloads: dict[str, list[dict[str, Any]]],
    provider_raw_payloads: dict[str, list[dict[str, Any]]],
    rejected_rows: list[dict[str, Any]],
    provider_failures: list[dict[str, Any]],
    raw_chainlink_rows: list[dict[str, Any]],
    provider_chainlink_rows: list[dict[str, Any]],
    chainlink_collection_report: dict[str, Any],
    chainlink_reason_codes: list[str],
    feature_enrichment_warning_reason_codes: list[str],
    feature_enrichment_post_market_close_candle_rejected_count: int,
    pending_feature_enrichment: bool,
    feature_enrichment_reason_codes: list[str],
    feature_enrichment_max_attempts: int,
) -> dict[str, Any]:
    market_count = len(raw_payloads["raw_polymarket_markets.jsonl"])
    reject_counts = _reject_counts(rejected_rows)
    provider_raw_markets = provider_raw_payloads[
        "raw_polymarket_markets.jsonl"
    ]
    market_identity_source_counts = Counter(
        str(row.get("market_identity_source_type") or "unknown")
        for row in provider_raw_markets
    )
    market_identity_fallback_reason_counts: Counter[str] = Counter()
    clob_revalidation_attempt_counts: Counter[str] = Counter()
    clob_revalidation_retry_reason_counts: Counter[str] = Counter()
    for row in provider_raw_markets:
        market_identity_fallback_reason_counts.update(
            str(reason)
            for reason in row.get(
                "market_identity_cache_fallback_reason_codes"
            )
            or []
        )
        revalidation = dict(
            row.get("market_identity_clob_revalidation") or {}
        )
        if revalidation:
            clob_revalidation_attempt_counts.update(
                [str(int(revalidation.get("attempt_count") or 0))]
            )
            clob_revalidation_retry_reason_counts.update(
                str(reason)
                for reason in revalidation.get("retry_reason_codes") or []
            )
    provider_raw_orderbooks = provider_raw_payloads[
        "raw_polymarket_orderbooks.jsonl"
    ]
    source_type_counts = Counter(
        str(row.get("orderbook_source_type") or "unknown")
        for row in provider_raw_orderbooks
    )
    fallback_reason_counts: Counter[str] = Counter()
    for row in provider_raw_orderbooks:
        fallback_reason_counts.update(
            str(reason)
            for reason in row.get("orderbook_fallback_reason_codes") or []
        )
    return {
        "schema_version": ASYNC_SETTLEMENT_SCHEMA_VERSION,
        "phase": PENDING_CAPTURE_PHASE,
        "run_id": config.run_id,
        "market_families": list(config.market_families),
        "pending_feature_enrichment": pending_feature_enrichment,
        "pending_resolution": market_count > 0 and not pending_feature_enrichment,
        "capture_status": (
            "pending_feature_enrichment"
            if pending_feature_enrichment
            else "blocked_fail_closed"
            if provider_failures or market_count == 0
            else "pending_resolution"
        ),
        "feature_enrichment_attempt_count": 0,
        "feature_enrichment_max_attempts": feature_enrichment_max_attempts,
        "feature_enrichment_recovered": False,
        "feature_enrichment_reason_codes": feature_enrichment_reason_codes,
        "feature_enrichment_warning_reason_codes": (
            feature_enrichment_warning_reason_codes
        ),
        "feature_enrichment_post_market_close_candle_rejected_count": (
            feature_enrichment_post_market_close_candle_rejected_count
        ),
        "resolution_provider_called": False,
        "raw_polymarket_market_count": market_count,
        "provider_raw_market_identity_source_type_distribution": dict(
            sorted(market_identity_source_counts.items())
        ),
        "market_identity_cache_fallback_market_count": sum(
            1
            for row in provider_raw_markets
            if row.get("market_identity_cache_fallback_used") is True
        ),
        "market_identity_cache_fallback_reason_distribution": dict(
            sorted(market_identity_fallback_reason_counts.items())
        ),
        "market_identity_cache_provenance_violation_count": sum(
            1
            for row in provider_raw_markets
            if row.get("market_identity_cache_fallback_used") is True
            and row.get("market_identity_cache_provenance_valid") is not True
        ),
        "market_identity_clob_revalidation_passed_count": sum(
            1
            for row in provider_raw_markets
            if row.get("market_identity_clob_revalidation_passed") is True
        ),
        "market_identity_clob_revalidation_retry_succeeded_market_count": sum(
            1
            for row in provider_raw_markets
            if int(
                dict(
                    row.get("market_identity_clob_revalidation") or {}
                ).get("attempt_count")
                or 0
            )
            > 1
            and row.get("market_identity_clob_revalidation_passed") is True
        ),
        "market_identity_clob_revalidation_attempt_distribution": dict(
            sorted(clob_revalidation_attempt_counts.items())
        ),
        "market_identity_clob_revalidation_retry_reason_distribution": dict(
            sorted(clob_revalidation_retry_reason_counts.items())
        ),
        "market_identity_clob_revalidation_identity_relaxation_count": sum(
            1
            for row in provider_raw_markets
            if dict(
                row.get("market_identity_clob_revalidation") or {}
            ).get("retry_policy_relaxed_identity_checks")
            is not False
            and row.get("market_identity_cache_fallback_used") is True
        ),
        "market_identity_live_orderbook_validation_required": True,
        "raw_orderbook_row_count": len(raw_payloads["raw_polymarket_orderbooks.jsonl"]),
        "provider_raw_orderbook_snapshot_count": len(
            provider_raw_orderbooks
        ),
        "training_sampled_orderbook_row_count": len(
            raw_payloads["raw_polymarket_orderbooks.jsonl"]
        ),
        "provider_raw_artifacts_preserved": True,
        "provider_raw_orderbook_source_type_distribution": dict(
            sorted(source_type_counts.items())
        ),
        "provider_raw_orderbook_rest_fallback_row_count": sum(
            1
            for row in provider_raw_orderbooks
            if row.get("orderbook_rest_fallback_used") is True
        ),
        "provider_raw_orderbook_fallback_reason_distribution": dict(
            sorted(fallback_reason_counts.items())
        ),
        "raw_trade_row_count": len(raw_payloads["raw_polymarket_trades.jsonl"]),
        "raw_btc_candle_row_count": len(raw_payloads["raw_binance_btcusdt_klines.jsonl"]),
        "raw_chainlink_price_row_count": len(raw_chainlink_rows),
        "provider_raw_chainlink_price_row_count": len(provider_chainlink_rows),
        "chainlink_timestamp_causality_violation_count": sum(
            1
            for row in raw_chainlink_rows
            if int(row.get("source_ts") or 0) > int(row.get("available_at_ts") or 0)
        ),
        "chainlink_capture_reason_codes": chainlink_reason_codes,
        "chainlink_rtds_price_stream_fresh": chainlink_collection_report.get(
            "price_stream_fresh"
        ),
        "chainlink_rtds_price_stream_stale": chainlink_collection_report.get(
            "price_stream_stale"
        ),
        "chainlink_rtds_stale_reconnect_seconds": chainlink_collection_report.get(
            "stale_reconnect_seconds"
        ),
        "chainlink_rtds_stale_reconnect_count": int(
            chainlink_collection_report.get("stale_reconnect_count") or 0
        ),
        "chainlink_rtds_last_price_row_received_at_ts": (
            chainlink_collection_report.get("last_price_row_received_at_ts")
        ),
        "chainlink_rtds_current_price_stream_staleness_ms": (
            chainlink_collection_report.get("current_price_stream_staleness_ms")
        ),
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
    provider_raw_payloads: dict[str, list[dict[str, Any]]],
    provider_raw_dir: Path,
    report: dict[str, Any],
    raw_chainlink_rows: list[dict[str, Any]],
    provider_chainlink_rows: list[dict[str, Any]],
    feature_enrichment_warning_reason_codes: list[str],
    feature_enrichment_post_market_close_candle_rejected_count: int,
    pending_feature_enrichment: bool,
    feature_enrichment_reason_codes: list[str],
    feature_enrichment_max_attempts: int,
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
        "provider_raw_artifact_hashes": {
            filename: _sha256_file(provider_raw_dir / filename)
            for filename in RAW_CORPUS_FILENAMES
            if (provider_raw_dir / filename).exists()
        },
        "provider_raw_artifact_row_counts": {
            filename: len(provider_raw_payloads[filename])
            for filename in RAW_CORPUS_FILENAMES
        },
        "chainlink_raw_artifact_sha256": _optional_sha256_file(
            config.raw_dir / CHAINLINK_RTDS_RAW_FILENAME
        ),
        "chainlink_raw_artifact_row_count": len(raw_chainlink_rows),
        "provider_chainlink_raw_artifact_sha256": _optional_sha256_file(
            provider_raw_dir / CHAINLINK_RTDS_RAW_FILENAME
        ),
        "provider_chainlink_raw_artifact_row_count": len(provider_chainlink_rows),
        "chainlink_collection_report_sha256": _optional_sha256_file(
            config.run_dir / CHAINLINK_RTDS_COLLECTION_REPORT_FILENAME
        ),
        "provider_raw_artifacts_preserved": True,
        "training_raw_is_validated_sampled_view": not pending_feature_enrichment,
        "pending_feature_enrichment": pending_feature_enrichment,
        "feature_enrichment_attempt_count": 0,
        "feature_enrichment_max_attempts": feature_enrichment_max_attempts,
        "feature_enrichment_recovered": False,
        "feature_enrichment_reason_codes": feature_enrichment_reason_codes,
        "feature_enrichment_warning_reason_codes": (
            feature_enrichment_warning_reason_codes
        ),
        "feature_enrichment_post_market_close_candle_rejected_count": (
            feature_enrichment_post_market_close_candle_rejected_count
        ),
        "pending_resolution": report["pending_resolution"],
        "capture_status": report["capture_status"],
        "resolution_provider_called": False,
        **safety_fields(),
    }


def _pending_finalization_report(
    *,
    config: PolymarketRealCorpusRecorderConfig,
    raw_payloads: dict[str, list[dict[str, Any]]],
    provider_raw_payloads: dict[str, list[dict[str, Any]]],
    rejected_rows: list[dict[str, Any]],
    phase2_result: Any,
    phase2_error: str | None,
    exported_training_corpus_dir: Path | None,
    round_artifact_evidence: dict[str, Any],
    raw_chainlink_rows: list[dict[str, Any]],
    provider_chainlink_rows: list[dict[str, Any]],
    chainlink_corpus_evidence: dict[str, Any],
    feature_enrichment_report: dict[str, Any],
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
        "pending_feature_enrichment": False,
        "pending_resolution": status == "pending_resolution",
        "feature_enrichment_recovered": bool(
            feature_enrichment_report.get("feature_enrichment_recovered")
        ),
        "feature_enrichment_attempt_count": int(
            feature_enrichment_report.get("feature_enrichment_attempt_count")
            or 0
        ),
        "feature_enrichment_source_distribution": dict(
            feature_enrichment_report.get(
                "feature_enrichment_source_distribution"
            )
            or {}
        ),
        "feature_enrichment_reason_codes": list(
            feature_enrichment_report.get("feature_enrichment_reason_codes")
            or []
        ),
        "feature_enrichment_warning_reason_codes": list(
            feature_enrichment_report.get(
                "feature_enrichment_warning_reason_codes"
            )
            or []
        ),
        "feature_enrichment_post_market_close_candle_rejected_count": int(
            feature_enrichment_report.get(
                "feature_enrichment_post_market_close_candle_rejected_count"
            )
            or 0
        ),
        "feature_enrichment_causality_validation_passed": (
            feature_enrichment_report.get(
                "feature_enrichment_causality_validation_passed"
            )
            is True
            if feature_enrichment_report
            else True
        ),
        "resolution_provider_called": market_count > 0,
        "raw_polymarket_market_count": market_count,
        "raw_orderbook_row_count": len(raw_payloads["raw_polymarket_orderbooks.jsonl"]),
        "provider_raw_orderbook_snapshot_count": len(
            provider_raw_payloads["raw_polymarket_orderbooks.jsonl"]
        ),
        "training_sampled_orderbook_row_count": len(
            raw_payloads["raw_polymarket_orderbooks.jsonl"]
        ),
        "provider_raw_resolution_row_count": len(
            provider_raw_payloads["raw_polymarket_resolutions.jsonl"]
        ),
        "provider_raw_artifacts_preserved": True,
        "raw_trade_row_count": len(raw_payloads["raw_polymarket_trades.jsonl"]),
        "raw_btc_candle_row_count": len(raw_payloads["raw_binance_btcusdt_klines.jsonl"]),
        "raw_chainlink_price_row_count": len(raw_chainlink_rows),
        "provider_raw_chainlink_price_row_count": len(provider_chainlink_rows),
        "chainlink_corpus_evidence": chainlink_corpus_evidence,
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
        **{
            key: round_artifact_evidence[key]
            for key in (
                "round_artifact_export_mode",
                "round_artifacts_written",
                "round_artifacts_newly_finalized",
                "training_raw_round_count",
                "paper_audit_round_count",
                "latest_run_summary_sha256",
            )
        },
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
    provider_raw_payloads: dict[str, list[dict[str, Any]]],
    provider_raw_dir: Path,
    report: dict[str, Any],
    phase2_result: Any,
    exported_training_corpus_dir: Path | None,
    round_artifact_evidence: dict[str, Any],
    raw_chainlink_rows: list[dict[str, Any]],
    provider_chainlink_rows: list[dict[str, Any]],
    chainlink_corpus_evidence: dict[str, Any],
    feature_enrichment_report: dict[str, Any],
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
        "provider_raw_artifact_hashes": {
            filename: _sha256_file(provider_raw_dir / filename)
            for filename in RAW_CORPUS_FILENAMES
            if (provider_raw_dir / filename).exists()
        },
        "provider_raw_artifact_row_counts": {
            filename: len(provider_raw_payloads[filename])
            for filename in RAW_CORPUS_FILENAMES
        },
        "chainlink_raw_artifact_sha256": _optional_sha256_file(
            config.raw_dir / CHAINLINK_RTDS_RAW_FILENAME
        ),
        "chainlink_raw_artifact_row_count": len(raw_chainlink_rows),
        "provider_chainlink_raw_artifact_sha256": _optional_sha256_file(
            provider_raw_dir / CHAINLINK_RTDS_RAW_FILENAME
        ),
        "provider_chainlink_raw_artifact_row_count": len(provider_chainlink_rows),
        "chainlink_collection_report_sha256": _optional_sha256_file(
            config.run_dir / CHAINLINK_RTDS_COLLECTION_REPORT_FILENAME
        ),
        "chainlink_corpus_evidence": chainlink_corpus_evidence,
        "provider_raw_artifacts_preserved": True,
        "training_raw_is_validated_sampled_view": True,
        "finalization_status": report["finalization_status"],
        "pending_feature_enrichment": False,
        "pending_resolution": report["pending_resolution"],
        "feature_enrichment_recovered": report[
            "feature_enrichment_recovered"
        ],
        "feature_enrichment_attempt_count": report[
            "feature_enrichment_attempt_count"
        ],
        "feature_enrichment_source_distribution": report[
            "feature_enrichment_source_distribution"
        ],
        "feature_enrichment_post_market_close_candle_rejected_count": report[
            "feature_enrichment_post_market_close_candle_rejected_count"
        ],
        "feature_enrichment_causality_validation_passed": report[
            "feature_enrichment_causality_validation_passed"
        ],
        "feature_enrichment_report_path": (
            str(
                config.run_dir
                / "pending_round_feature_enrichment_report.json"
            )
            if feature_enrichment_report
            else None
        ),
        "feature_enrichment_report_sha256": (
            _optional_sha256_file(
                config.run_dir
                / "pending_round_feature_enrichment_report.json"
            )
            if feature_enrichment_report
            else None
        ),
        "feature_enrichment_manifest_path": (
            str(
                config.run_dir
                / "pending_round_feature_enrichment_manifest.json"
            )
            if feature_enrichment_report
            else None
        ),
        "feature_enrichment_manifest_sha256": (
            _optional_sha256_file(
                config.run_dir
                / "pending_round_feature_enrichment_manifest.json"
            )
            if feature_enrichment_report
            else None
        ),
        "resolution_provider_called": report["resolution_provider_called"],
        "phase2_corpus_built": phase2_result is not None,
        "phase2_corpus_dir": None if phase2_result is None else str(phase2_result.output_dir),
        "phase2_corpus_manifest_sha256": report["phase2_corpus_manifest_sha256"],
        "round_artifact_export_mode": round_artifact_evidence[
            "round_artifact_export_mode"
        ],
        "round_artifacts_written": round_artifact_evidence["round_artifacts_written"],
        "round_artifacts_newly_finalized": round_artifact_evidence[
            "round_artifacts_newly_finalized"
        ],
        "training_raw_round_count": round_artifact_evidence["training_raw_round_count"],
        "paper_audit_round_count": round_artifact_evidence["paper_audit_round_count"],
        "latest_run_summary_sha256": round_artifact_evidence[
            "latest_run_summary_sha256"
        ],
        "round_lifecycle_index_paths": round_artifact_evidence[
            "round_lifecycle_index_paths"
        ],
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


def _provider_raw_payloads(
    *,
    market_rows: list[dict[str, Any]],
    orderbook_rows: list[dict[str, Any]],
    trade_rows: list[dict[str, Any]],
    btc_candle_rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    payloads = empty_raw_payloads()
    payloads["raw_polymarket_markets.jsonl"] = [dict(row) for row in market_rows]
    payloads["raw_polymarket_orderbooks.jsonl"] = [
        dict(row) for row in orderbook_rows
    ]
    payloads["raw_polymarket_trades.jsonl"] = [dict(row) for row in trade_rows]
    payloads["raw_binance_btcusdt_klines.jsonl"] = [
        dict(row) for row in btc_candle_rows
    ]
    return payloads


def _write_raw_files(raw_dir: Path, raw_payloads: dict[str, list[dict[str, Any]]]) -> None:
    for filename in RAW_CORPUS_FILENAMES:
        _write_jsonl(raw_dir / filename, raw_payloads[filename])


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


def _reject_counts(rejected_rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for row in rejected_rows:
        for reason in row.get("reject_reasons", []):
            counts[str(reason)] += 1
    return dict(sorted(counts.items()))


def _feature_enrichment_can_retry(
    *,
    accepted_markets: list[dict[str, Any]],
    candle_reasons: list[str],
    feature_provider_failures: list[dict[str, Any]],
) -> bool:
    if not accepted_markets or set(candle_reasons) != {
        "missing_btc_feature_candles"
    }:
        return False
    allowed_reason_codes = {
        "btc_feature_candle_sources_unavailable",
        "read_only_public_http_timeout",
        "read_only_public_http_transport_error",
        "read_only_public_http_server_error",
    }
    observed_reason_codes = {
        str(reason)
        for row in feature_provider_failures
        for reason in row.get("reject_reasons", [])
    }
    return not observed_reason_codes or observed_reason_codes <= (
        allowed_reason_codes
    )


def _causal_feature_candles_for_markets(
    *,
    rows: list[dict[str, Any]],
    markets: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, int]:
    max_market_end_ts = max(
        (int(row.get("market_end_ts") or 0) for row in markets),
        default=0,
    )
    if max_market_end_ts <= 0:
        return [], len(rows), max_market_end_ts
    accepted = [
        row
        for row in rows
        if int(
            row.get("available_at_ts")
            or row.get("close_time")
            or 0
        )
        <= max_market_end_ts
    ]
    return accepted, len(rows) - len(accepted), max_market_end_ts


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0.0 else None


def _optional_sha256_file(path: Path) -> str | None:
    return _sha256_file(path) if path.exists() else None


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
