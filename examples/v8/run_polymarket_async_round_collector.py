"""Run non-blocking round capture with asynchronous settlement finalization."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket import (  # noqa: E402
    DEFAULT_POLYMARKET_CLOB_WS_MARKET_URL,
    DEFAULT_POLYMARKET_RTDS_URL,
    V8_TRAINING_CORPUS_ROOT,
    PolymarketChainlinkRTDSCollector,
    PolymarketPublicHTTPRealCorpusProvider,
    PolymarketRealCorpusRecorderConfig,
    capture_polymarket_pending_round,
    finalize_polymarket_pending_round,
)
from bigan.v8.polymarket.corpus import BTC_UPDOWN_MARKET_HORIZONS_MS  # noqa: E402
from bigan.v8.polymarket.training.execution_layer_v2_pairwise_action_advantage_lcb import (  # noqa: E402
    _capture_quality_audit,
    _load_json,
)
from bigan.v8.polymarket.training.execution_layer_v2_pairwise_future_unseen_holdout import (  # noqa: E402
    MAXIMUM_CAPTURE_ATTEMPT_COUNT,
    TARGET_VALID_MARKET_COUNT,
    load_and_validate_pairwise_future_unseen_collection_freeze,
)


def run_polymarket_async_round_collector_cli(
    *,
    batch_id: str,
    output_dir: Path | str,
    round_count: int,
    market_family: str = "btc_updown_5m",
    public_provider_timeout_seconds: float = 330.0,
    public_provider_http_timeout_seconds: float = 15.0,
    orderbook_snapshot_interval_seconds: float = 1.0,
    orderbook_ws_initial_complete_book_timeout_seconds: float = 15.0,
    rest_orderbook_fallback_collection_seconds: float = 330.0,
    settlement_poll_interval_seconds: float = 15.0,
    settlement_grace_seconds: float = 0.0,
    training_corpus_root: Path | str = V8_TRAINING_CORPUS_ROOT,
    clob_ws_url: str = DEFAULT_POLYMARKET_CLOB_WS_MARKET_URL,
    max_round_start_lag_seconds: float = 30.0,
    chainlink_rtds_url: str = DEFAULT_POLYMARKET_RTDS_URL,
    chainlink_rtds_warmup_seconds: float = 5.0,
    chainlink_rtds_stale_reconnect_seconds: float = 15.0,
    market_identity_cache_path: Path | str | None = None,
    gamma_market_identity_prefetch_round_count: int = 12,
    market_identity_cache_max_age_seconds: float = 7_200.0,
    clob_identity_revalidation_max_attempts: int = 3,
    clob_identity_revalidation_retry_seconds: float = 0.25,
    feature_enrichment_max_attempts: int = 40,
    outcome_blind_quality_stop_target: int | None = None,
    outcome_blind_collection_only: bool = False,
    future_holdout_collection_freeze_manifest: Path | str | None = None,
    future_holdout_collection_freeze_manifest_sha256: str | None = None,
    overwrite_existing: bool = False,
) -> dict[str, Any]:
    if round_count <= 0:
        raise ValueError("round_count must be positive")
    if public_provider_timeout_seconds <= 0:
        raise ValueError("public_provider_timeout_seconds must be positive")
    if public_provider_http_timeout_seconds <= 0:
        raise ValueError("public_provider_http_timeout_seconds must be positive")
    if orderbook_ws_initial_complete_book_timeout_seconds <= 0:
        raise ValueError(
            "orderbook_ws_initial_complete_book_timeout_seconds must be positive"
        )
    if rest_orderbook_fallback_collection_seconds <= 0:
        raise ValueError(
            "rest_orderbook_fallback_collection_seconds must be positive"
        )
    if settlement_poll_interval_seconds <= 0:
        raise ValueError("settlement_poll_interval_seconds must be positive")
    if settlement_grace_seconds < 0:
        raise ValueError("settlement_grace_seconds must be non-negative")
    if max_round_start_lag_seconds < 0:
        raise ValueError("max_round_start_lag_seconds must be non-negative")
    if chainlink_rtds_warmup_seconds < 0:
        raise ValueError("chainlink_rtds_warmup_seconds must be non-negative")
    if chainlink_rtds_stale_reconnect_seconds <= 0:
        raise ValueError("chainlink_rtds_stale_reconnect_seconds must be positive")
    if gamma_market_identity_prefetch_round_count < 0:
        raise ValueError(
            "gamma_market_identity_prefetch_round_count must be non-negative"
        )
    if market_identity_cache_max_age_seconds <= 0:
        raise ValueError("market_identity_cache_max_age_seconds must be positive")
    if clob_identity_revalidation_max_attempts <= 0:
        raise ValueError(
            "clob_identity_revalidation_max_attempts must be positive"
        )
    if clob_identity_revalidation_retry_seconds < 0:
        raise ValueError(
            "clob_identity_revalidation_retry_seconds must be non-negative"
        )
    if feature_enrichment_max_attempts <= 0:
        raise ValueError("feature_enrichment_max_attempts must be positive")

    quality_control_state: dict[str, Any] | None = None
    quality_collector_contract: dict[str, Any] | None = None
    minimum_collection_decision_ts: int | None = None
    if outcome_blind_quality_stop_target is not None:
        if outcome_blind_quality_stop_target != TARGET_VALID_MARKET_COUNT:
            raise ValueError(
                "outcome_blind_quality_stop_target must match the frozen target"
            )
        if round_count != MAXIMUM_CAPTURE_ATTEMPT_COUNT:
            raise ValueError("round_count must match the frozen maximum attempt count")
        if (
            future_holdout_collection_freeze_manifest is None
            or not future_holdout_collection_freeze_manifest_sha256
        ):
            raise ValueError(
                "future holdout collection freeze manifest and SHA-256 are required"
            )
        if not outcome_blind_collection_only:
            raise ValueError(
                "outcome_blind_collection_only is required for frozen future holdout collection"
            )
        if settlement_grace_seconds != 0.0:
            raise ValueError(
                "settlement_grace_seconds must be zero in outcome-blind collection-only mode"
            )
        collection_freeze_path = Path(
            future_holdout_collection_freeze_manifest
        ).expanduser().resolve()
        collection_freeze, collection_freeze_audit = (
            load_and_validate_pairwise_future_unseen_collection_freeze(
                collection_freeze_path,
                future_holdout_collection_freeze_manifest_sha256,
            )
        )
        minimum_collection_decision_ts = int(
            collection_freeze_audit["minimum_collection_decision_ts"]
        )
        candidate_protocol = _load_json(
            Path(str(collection_freeze["candidate_protocol"]["path"]))
        )
        quality_collector_contract = dict(candidate_protocol["collector_contract"])
        quality_control_state = {
            "outcome_blind_quality_stop_enabled": True,
            "outcome_blind_collection_only": True,
            "outcome_blind_quality_stop_target": TARGET_VALID_MARKET_COUNT,
            "maximum_capture_attempt_count": MAXIMUM_CAPTURE_ATTEMPT_COUNT,
            "future_holdout_collection_freeze_manifest": {
                "path": str(collection_freeze_path),
                "sha256": future_holdout_collection_freeze_manifest_sha256.lower(),
            },
            "minimum_collection_decision_ts": minimum_collection_decision_ts,
            "strictly_later_than_source_boundary_enforced": True,
            "quality_stop_inputs": "capture_quality_and_provenance_only",
            "uses_model_scores_for_collection_control": False,
            "uses_accepted_bet_count_for_collection_control": False,
            "labels_or_outcomes_opened_for_collection_control": False,
            "labels_or_outcomes_opened_during_collection": False,
            "settlement_pnl_opened_for_collection_control": False,
            "settlement_finalizer_started": False,
            "resolution_provider_called": False,
            "training_corpus_export_attempted": False,
            "outcome_blind_quality_valid_capture_count": 0,
            "outcome_blind_quality_valid_capture_run_ids": [],
            "outcome_blind_quality_excluded_capture_count": 0,
            "outcome_blind_quality_exclusion_reason_distribution": {},
            "quality_target_reached": False,
            "collection_stop_reason": "frozen_maximum_not_yet_reached",
        }

    root = Path(output_dir).expanduser().resolve()
    resolved_market_identity_cache_path = (
        root / "gamma_market_identity_cache.json"
        if market_identity_cache_path is None
        else Path(market_identity_cache_path).expanduser().resolve()
    )
    batch_dir = root / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    stop_event = threading.Event()
    lock = threading.Lock()
    captures: list[dict[str, Any]] = []
    finalizations: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    capture_threads: list[threading.Thread] = []
    chainlink_collector = PolymarketChainlinkRTDSCollector(
        url=chainlink_rtds_url,
        stale_reconnect_seconds=chainlink_rtds_stale_reconnect_seconds,
    )
    chainlink_collector.start()
    if chainlink_rtds_warmup_seconds:
        chainlink_collector.wait_for_rows(timeout_seconds=chainlink_rtds_warmup_seconds)

    def finalizer_loop() -> None:
        while not stop_event.is_set():
            _finalize_pending_once(
                output_dir=root,
                destination_root=Path(training_corpus_root),
                clob_ws_url=clob_ws_url,
                overwrite_existing=overwrite_existing,
                batch_id_prefix=batch_id,
                finalizations=finalizations,
                errors=errors,
                lock=lock,
                captures=captures,
                public_provider_http_timeout_seconds=(
                    public_provider_http_timeout_seconds
                ),
            )
            stop_event.wait(settlement_poll_interval_seconds)

    finalizer: threading.Thread | None = None
    if _settlement_finalization_permitted(
        outcome_blind_collection_only=outcome_blind_collection_only
    ):
        finalizer = threading.Thread(
            target=finalizer_loop,
            name=f"{batch_id}-settlement-finalizer",
            daemon=True,
        )
        finalizer.start()

    def capture_round(
        *,
        index: int,
        run_id: str,
        scheduled_round_start_epoch_seconds: float,
    ) -> None:
        capture_started_epoch_seconds = _wait_until_scheduled_round_start(
            scheduled_round_start_epoch_seconds
        )
        try:
            provider = PolymarketPublicHTTPRealCorpusProvider(
                max_markets=1,
                clob_ws_url=clob_ws_url,
                timeout_seconds=public_provider_timeout_seconds,
                http_timeout_seconds=public_provider_http_timeout_seconds,
                orderbook_snapshot_interval_seconds=orderbook_snapshot_interval_seconds,
                orderbook_ws_initial_complete_book_timeout_seconds=(
                    orderbook_ws_initial_complete_book_timeout_seconds
                ),
                rest_fallback_collection_seconds=(
                    rest_orderbook_fallback_collection_seconds
                ),
                market_identity_cache_path=resolved_market_identity_cache_path,
                market_identity_cache_max_age_seconds=(
                    market_identity_cache_max_age_seconds
                ),
                gamma_market_identity_prefetch_round_count=(
                    gamma_market_identity_prefetch_round_count
                ),
                clob_identity_revalidation_max_attempts=(
                    clob_identity_revalidation_max_attempts
                ),
                clob_identity_revalidation_retry_seconds=(
                    clob_identity_revalidation_retry_seconds
                ),
            )
            config = PolymarketRealCorpusRecorderConfig(
                run_id=run_id,
                output_dir=root,
                market_families=(market_family,),
                mock_public_data=False,
                overwrite_existing=overwrite_existing,
            )
            capture = capture_polymarket_pending_round(
                config,
                public_provider=provider,
                chainlink_rtds_collector=chainlink_collector,
                feature_enrichment_max_attempts=(
                    feature_enrichment_max_attempts
                ),
            )
            market_identity_cache_report = (
                provider.market_identity_cache_report()
            )
        except Exception as exc:  # noqa: BLE001
            with lock:
                errors.append(
                    {
                        "round_index": index,
                        "run_id": run_id,
                        "run_dir": str(root / run_id),
                        "scheduled_round_start_ts": int(
                            scheduled_round_start_epoch_seconds * 1000
                        ),
                        "capture_thread_started_at_ts": int(
                            capture_started_epoch_seconds * 1000
                        ),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "stage": "round_capture",
                    }
                )
                _write_json(
                    batch_dir / "batch_progress.json",
                    _summary(
                        batch_id,
                        captures,
                        finalizations,
                        errors,
                        quality_control_state=quality_control_state,
                    ),
                )
            return
        with lock:
            captures.append(
                {
                    "round_index": index,
                    "run_id": run_id,
                    "run_dir": str(capture.run_dir),
                    "scheduled_round_start_ts": int(scheduled_round_start_epoch_seconds * 1000),
                    "capture_thread_started_at_ts": int(capture_started_epoch_seconds * 1000),
                    "capture_start_lag_seconds": (
                        capture_started_epoch_seconds - scheduled_round_start_epoch_seconds
                    ),
                    "capture_start_boundary_validation_passed": (
                        capture_started_epoch_seconds >= scheduled_round_start_epoch_seconds
                    ),
                    "capture_status": capture.report["capture_status"],
                    "pending_feature_enrichment": capture.report[
                        "pending_feature_enrichment"
                    ],
                    "pending_resolution": capture.report["pending_resolution"],
                    "feature_enrichment_attempt_count": capture.report[
                        "feature_enrichment_attempt_count"
                    ],
                    "feature_enrichment_max_attempts": capture.report[
                        "feature_enrichment_max_attempts"
                    ],
                    "feature_enrichment_recovered": capture.report[
                        "feature_enrichment_recovered"
                    ],
                    "feature_enrichment_reason_codes": capture.report[
                        "feature_enrichment_reason_codes"
                    ],
                    "feature_enrichment_warning_reason_codes": (
                        capture.report[
                            "feature_enrichment_warning_reason_codes"
                        ]
                    ),
                    "feature_enrichment_post_market_close_candle_rejected_count": (
                        capture.report[
                            "feature_enrichment_post_market_close_candle_rejected_count"
                        ]
                    ),
                    "market_family": market_family,
                    "orderbook_snapshot_interval_seconds": (
                        orderbook_snapshot_interval_seconds
                    ),
                    "public_provider_timeout_seconds": (
                        public_provider_timeout_seconds
                    ),
                    "public_provider_http_timeout_seconds": (
                        public_provider_http_timeout_seconds
                    ),
                    "orderbook_ws_initial_complete_book_timeout_seconds": (
                        orderbook_ws_initial_complete_book_timeout_seconds
                    ),
                    "rest_orderbook_fallback_collection_seconds": (
                        rest_orderbook_fallback_collection_seconds
                    ),
                    "rest_orderbook_fallback_stops_at_market_close": True,
                    "market_identity_cache_path": str(
                        resolved_market_identity_cache_path
                    ),
                    "market_identity_cache_max_age_seconds": (
                        market_identity_cache_max_age_seconds
                    ),
                    "gamma_market_identity_prefetch_round_count": (
                        gamma_market_identity_prefetch_round_count
                    ),
                    "clob_identity_revalidation_max_attempts": (
                        clob_identity_revalidation_max_attempts
                    ),
                    "clob_identity_revalidation_retry_seconds": (
                        clob_identity_revalidation_retry_seconds
                    ),
                    "market_identity_cache_report": (
                        market_identity_cache_report
                    ),
                    "raw_polymarket_market_count": capture.report["raw_polymarket_market_count"],
                    "provider_raw_market_identity_source_type_distribution": (
                        capture.report[
                            "provider_raw_market_identity_source_type_distribution"
                        ]
                    ),
                    "market_identity_cache_fallback_market_count": (
                        capture.report[
                            "market_identity_cache_fallback_market_count"
                        ]
                    ),
                    "market_identity_cache_fallback_reason_distribution": (
                        capture.report[
                            "market_identity_cache_fallback_reason_distribution"
                        ]
                    ),
                    "market_identity_cache_provenance_violation_count": (
                        capture.report[
                            "market_identity_cache_provenance_violation_count"
                        ]
                    ),
                    "market_identity_clob_revalidation_passed_count": (
                        capture.report[
                            "market_identity_clob_revalidation_passed_count"
                        ]
                    ),
                    "market_identity_clob_revalidation_retry_succeeded_market_count": (
                        capture.report[
                            "market_identity_clob_revalidation_retry_succeeded_market_count"
                        ]
                    ),
                    "market_identity_clob_revalidation_attempt_distribution": (
                        capture.report[
                            "market_identity_clob_revalidation_attempt_distribution"
                        ]
                    ),
                    "market_identity_clob_revalidation_retry_reason_distribution": (
                        capture.report[
                            "market_identity_clob_revalidation_retry_reason_distribution"
                        ]
                    ),
                    "market_identity_clob_revalidation_identity_relaxation_count": (
                        capture.report[
                            "market_identity_clob_revalidation_identity_relaxation_count"
                        ]
                    ),
                    "raw_orderbook_row_count": capture.report["raw_orderbook_row_count"],
                    "provider_raw_orderbook_snapshot_count": capture.report[
                        "provider_raw_orderbook_snapshot_count"
                    ],
                    "training_sampled_orderbook_row_count": capture.report[
                        "training_sampled_orderbook_row_count"
                    ],
                    "provider_raw_orderbook_source_type_distribution": (
                        capture.report["provider_raw_orderbook_source_type_distribution"]
                    ),
                    "provider_raw_orderbook_rest_fallback_row_count": (
                        capture.report["provider_raw_orderbook_rest_fallback_row_count"]
                    ),
                    "provider_raw_orderbook_fallback_reason_distribution": (
                        capture.report["provider_raw_orderbook_fallback_reason_distribution"]
                    ),
                    "raw_trade_row_count": capture.report["raw_trade_row_count"],
                    "raw_btc_candle_row_count": capture.report["raw_btc_candle_row_count"],
                    "raw_chainlink_price_row_count": capture.report[
                        "raw_chainlink_price_row_count"
                    ],
                    "chainlink_capture_reason_codes": capture.report[
                        "chainlink_capture_reason_codes"
                    ],
                    "chainlink_rtds_price_stream_fresh": capture.report[
                        "chainlink_rtds_price_stream_fresh"
                    ],
                    "chainlink_rtds_price_stream_stale": capture.report[
                        "chainlink_rtds_price_stream_stale"
                    ],
                    "chainlink_rtds_stale_reconnect_seconds": capture.report[
                        "chainlink_rtds_stale_reconnect_seconds"
                    ],
                    "chainlink_rtds_stale_reconnect_count": capture.report[
                        "chainlink_rtds_stale_reconnect_count"
                    ],
                    "chainlink_rtds_last_price_row_received_at_ts": capture.report[
                        "chainlink_rtds_last_price_row_received_at_ts"
                    ],
                    "chainlink_rtds_current_price_stream_staleness_ms": (
                        capture.report[
                            "chainlink_rtds_current_price_stream_staleness_ms"
                        ]
                    ),
                    "reject_reason_counts": capture.report["reject_reason_counts"],
                }
            )
            captures.sort(key=lambda item: int(item.get("round_index") or 0))
            if quality_control_state is not None:
                quality_control_state.update(
                    _outcome_blind_quality_control_snapshot(
                        captures,
                        collector_contract=quality_collector_contract or {},
                        target_count=outcome_blind_quality_stop_target or 0,
                    )
                )
            _write_json(
                batch_dir / "batch_progress.json",
                _summary(
                    batch_id,
                    captures,
                    finalizations,
                    errors,
                    quality_control_state=quality_control_state,
                ),
            )

    try:
        previous_round_start_epoch_seconds: float | None = None
        for index in range(1, round_count + 1):
            scheduled_round_start_epoch_seconds = _sleep_until_round_start_window(
                market_family=market_family,
                max_round_start_lag_seconds=max_round_start_lag_seconds,
                previous_round_start_epoch_seconds=(previous_round_start_epoch_seconds),
            )
            while (
                minimum_collection_decision_ts is not None
                and int(scheduled_round_start_epoch_seconds * 1000)
                < minimum_collection_decision_ts
            ):
                scheduled_round_start_epoch_seconds = _sleep_until_round_start_window(
                    market_family=market_family,
                    max_round_start_lag_seconds=max_round_start_lag_seconds,
                    previous_round_start_epoch_seconds=(
                        scheduled_round_start_epoch_seconds
                    ),
                )
            if quality_control_state is not None:
                with lock:
                    quality_control_state.update(
                        _outcome_blind_quality_control_snapshot(
                            captures,
                            collector_contract=quality_collector_contract or {},
                            target_count=outcome_blind_quality_stop_target or 0,
                        )
                    )
                    if quality_control_state["quality_target_reached"]:
                        quality_control_state["collection_stop_reason"] = (
                            "outcome_blind_quality_target_reached"
                        )
                        _write_json(
                            batch_dir / "batch_progress.json",
                            _summary(
                                batch_id,
                                captures,
                                finalizations,
                                errors,
                                quality_control_state=quality_control_state,
                            ),
                        )
                        break
            run_id = f"{batch_id}-round{index:02d}-{_utc_stamp()}"
            capture_thread = threading.Thread(
                target=capture_round,
                kwargs={
                    "index": index,
                    "run_id": run_id,
                    "scheduled_round_start_epoch_seconds": (scheduled_round_start_epoch_seconds),
                },
                name=f"{batch_id}-capture-{index:04d}",
                daemon=True,
            )
            capture_thread.start()
            capture_threads.append(capture_thread)
            previous_round_start_epoch_seconds = scheduled_round_start_epoch_seconds
        for capture_thread in capture_threads:
            capture_thread.join()
        if settlement_grace_seconds:
            deadline = time.monotonic() + settlement_grace_seconds
            while time.monotonic() < deadline:
                time.sleep(min(settlement_poll_interval_seconds, deadline - time.monotonic()))
    finally:
        stop_event.set()
        if finalizer is not None:
            finalizer.join(timeout=max(1.0, settlement_poll_interval_seconds))
        chainlink_collector.stop()

    if _settlement_finalization_permitted(
        outcome_blind_collection_only=outcome_blind_collection_only
    ):
        _finalize_pending_once(
            output_dir=root,
            destination_root=Path(training_corpus_root),
            clob_ws_url=clob_ws_url,
            overwrite_existing=overwrite_existing,
            batch_id_prefix=batch_id,
            finalizations=finalizations,
            errors=errors,
            lock=lock,
            captures=captures,
            public_provider_http_timeout_seconds=(
                public_provider_http_timeout_seconds
            ),
        )
    if quality_control_state is not None:
        quality_control_state.update(
            _outcome_blind_quality_control_snapshot(
                captures,
                collector_contract=quality_collector_contract or {},
                target_count=outcome_blind_quality_stop_target or 0,
            )
        )
        if quality_control_state["quality_target_reached"]:
            quality_control_state["collection_stop_reason"] = (
                "outcome_blind_quality_target_reached"
            )
        else:
            quality_control_state["collection_stop_reason"] = (
                "frozen_maximum_capture_attempt_count_reached_without_target"
            )
    summary = _summary(
        batch_id,
        captures,
        finalizations,
        errors,
        quality_control_state=quality_control_state,
    )
    summary["chainlink_rtds_collection_report"] = chainlink_collector.collection_report()
    summary["outcome_blind_collection_only"] = outcome_blind_collection_only
    summary["settlement_finalizer_started"] = finalizer is not None
    summary["resolution_provider_called"] = finalizer is not None
    summary["training_corpus_export_attempted"] = finalizer is not None
    summary["labels_or_outcomes_opened_during_collection"] = (
        False if outcome_blind_collection_only else None
    )
    summary["settlement_pnl_opened_during_collection"] = (
        False if outcome_blind_collection_only else None
    )
    summary_path = batch_dir / "batch_summary.json"
    _write_json(summary_path, summary)
    summary["batch_summary_path"] = str(summary_path)
    return summary


def _settlement_finalization_permitted(
    *, outcome_blind_collection_only: bool
) -> bool:
    """Keep future-holdout outcomes sealed until explicit candidate binding."""

    return not outcome_blind_collection_only


def run_polymarket_async_finalizer_cli(
    *,
    batch_id: str,
    output_dir: Path | str,
    settlement_poll_interval_seconds: float = 15.0,
    settlement_grace_seconds: float = 0.0,
    training_corpus_root: Path | str = V8_TRAINING_CORPUS_ROOT,
    clob_ws_url: str = DEFAULT_POLYMARKET_CLOB_WS_MARKET_URL,
    public_provider_http_timeout_seconds: float = 15.0,
    overwrite_existing: bool = False,
) -> dict[str, Any]:
    if settlement_poll_interval_seconds <= 0:
        raise ValueError("settlement_poll_interval_seconds must be positive")
    if settlement_grace_seconds < 0:
        raise ValueError("settlement_grace_seconds must be non-negative")
    if public_provider_http_timeout_seconds <= 0:
        raise ValueError("public_provider_http_timeout_seconds must be positive")
    root = Path(output_dir).expanduser().resolve()
    batch_dir = root / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    progress_path = batch_dir / "batch_progress.json"
    lock = threading.Lock()
    if progress_path.is_file():
        previous = _read_json(progress_path)
        if previous.get("batch_id") != batch_id:
            raise ValueError("existing batch progress identity mismatch")
        if previous.get("paper_only") is not True or previous.get("capital_at_risk") is not False:
            raise ValueError("existing batch progress safety contract failed")
        captures = [dict(row) for row in previous.get("captures") or []]
        finalizations = [dict(row) for row in previous.get("finalizations") or []]
        errors = [dict(row) for row in previous.get("errors") or []]
    else:
        captures = []
        finalizations = []
        errors = []
    deadline = time.monotonic() + settlement_grace_seconds
    while True:
        _finalize_pending_once(
            output_dir=root,
            destination_root=Path(training_corpus_root),
            clob_ws_url=clob_ws_url,
            overwrite_existing=overwrite_existing,
            batch_id_prefix=batch_id,
            finalizations=finalizations,
            errors=errors,
            lock=lock,
            captures=captures,
            public_provider_http_timeout_seconds=(
                public_provider_http_timeout_seconds
            ),
        )
        _write_json_atomic(
            progress_path,
            _summary(batch_id, captures, finalizations, errors),
        )
        if settlement_grace_seconds <= 0 or time.monotonic() >= deadline:
            break
        time.sleep(min(settlement_poll_interval_seconds, deadline - time.monotonic()))
    summary = {
        "batch_id": batch_id,
        "paper_only": True,
        "capital_at_risk": False,
        "finalize_only": True,
        "finalization_attempt_count": len(finalizations),
        "exported_round_count": sum(
            1 for item in finalizations if item.get("finalization_status") == "exported"
        ),
        "pending_resolution_count": sum(
            1 for item in finalizations if item.get("finalization_status") == "pending_resolution"
        ),
        "pending_feature_enrichment_count": sum(
            1
            for item in finalizations
            if item.get("finalization_status") == "pending_feature_enrichment"
        ),
        "feature_enrichment_recovered_count": sum(
            1
            for item in finalizations
            if item.get("feature_enrichment_recovered") is True
        ),
        "error_count": len(errors),
        "finalizations": finalizations,
        "errors": errors,
    }
    summary_path = batch_dir / "finalizer_summary.json"
    _write_json(summary_path, summary)
    summary["finalizer_summary_path"] = str(summary_path)
    return summary


def _finalize_pending_once(
    *,
    output_dir: Path,
    destination_root: Path,
    clob_ws_url: str,
    overwrite_existing: bool,
    batch_id_prefix: str | None,
    finalizations: list[dict[str, Any]],
    errors: list[dict[str, str]],
    lock: threading.Lock,
    captures: list[dict[str, Any]] | None = None,
    public_provider_http_timeout_seconds: float = 15.0,
) -> None:
    seen_exported = {
        item["run_id"] for item in finalizations if item.get("finalization_status") == "exported"
    }
    for manifest_path in sorted(output_dir.glob("*/pending_round_capture_manifest.json")):
        run_dir = manifest_path.parent
        if batch_id_prefix is not None and not run_dir.name.startswith(f"{batch_id_prefix}-round"):
            continue
        if run_dir.name in seen_exported:
            continue
        capture_manifest = _read_json(manifest_path)
        if not (
            capture_manifest.get("pending_resolution")
            or capture_manifest.get("pending_feature_enrichment")
        ):
            continue
        finalization_report_path = run_dir / "pending_round_finalization_report.json"
        if finalization_report_path.exists():
            previous = _read_json(finalization_report_path)
            if previous.get("finalization_status") == "exported":
                try:
                    recovered = _recover_existing_exported_finalization(
                        run_dir=run_dir,
                        destination_root=destination_root,
                        report=previous,
                    )
                except ValueError as exc:
                    with lock:
                        errors.append(
                            {
                                "run_dir": str(run_dir),
                                "error": str(exc),
                                "stage": "existing_exported_finalization_recovery",
                            }
                        )
                    continue
                with lock:
                    _upsert_by_run_id(finalizations, recovered)
                continue
        try:
            provider = PolymarketPublicHTTPRealCorpusProvider(
                max_markets=1,
                clob_ws_url=clob_ws_url,
                timeout_seconds=15.0,
                http_timeout_seconds=public_provider_http_timeout_seconds,
            )
            result = finalize_polymarket_pending_round(
                run_dir,
                public_provider=provider,
                destination_root=destination_root,
                overwrite_existing=overwrite_existing,
            )
        except Exception as exc:  # noqa: BLE001
            with lock:
                errors.append({"run_dir": str(run_dir), "error": str(exc)})
            continue
        with lock:
            if captures is not None:
                _apply_feature_enrichment_result_to_capture(
                    captures,
                    run_id=run_dir.name,
                    report=result.report,
                )
            _upsert_by_run_id(
                finalizations,
                {
                    "run_id": run_dir.name,
                    "run_dir": str(run_dir),
                    "finalization_status": result.report["finalization_status"],
                    "pending_feature_enrichment": result.report.get(
                        "pending_feature_enrichment",
                        False,
                    ),
                    "pending_resolution": result.report["pending_resolution"],
                    "feature_enrichment_attempt_count": int(
                        result.report.get("feature_enrichment_attempt_count")
                        or 0
                    ),
                    "feature_enrichment_recovered": (
                        result.report.get("feature_enrichment_recovered")
                        is True
                    ),
                    "feature_enrichment_reason_codes": list(
                        result.report.get("feature_enrichment_reason_codes")
                        or []
                    ),
                    "feature_enrichment_warning_reason_codes": list(
                        result.report.get(
                            "feature_enrichment_warning_reason_codes"
                        )
                        or []
                    ),
                    "feature_enrichment_post_market_close_candle_rejected_count": int(
                        result.report.get(
                            "feature_enrichment_post_market_close_candle_rejected_count"
                        )
                        or 0
                    ),
                    "raw_btc_candle_row_count": int(
                        result.report.get("raw_btc_candle_row_count") or 0
                    ),
                    "training_eligible": result.report["training_eligible"],
                    "exported_training_corpus_dir": result.report["exported_training_corpus_dir"],
                    "raw_resolution_count": result.report["raw_resolution_count"],
                    "reject_reason_counts": result.report["reject_reason_counts"],
                },
            )


def _apply_feature_enrichment_result_to_capture(
    captures: list[dict[str, Any]],
    *,
    run_id: str,
    report: dict[str, Any],
) -> None:
    for capture in captures:
        if capture.get("run_id") != run_id:
            continue
        capture.update(
            {
                "capture_status": (
                    "pending_resolution"
                    if report.get("feature_enrichment_recovered") is True
                    else report.get("finalization_status")
                ),
                "pending_feature_enrichment": report.get(
                    "pending_feature_enrichment",
                    False,
                ),
                "pending_resolution": report.get("pending_resolution") is True,
                "feature_enrichment_attempt_count": int(
                    report.get("feature_enrichment_attempt_count") or 0
                ),
                "feature_enrichment_recovered": (
                    report.get("feature_enrichment_recovered") is True
                ),
                "feature_enrichment_reason_codes": list(
                    report.get("feature_enrichment_reason_codes") or []
                ),
                "feature_enrichment_warning_reason_codes": list(
                    report.get("feature_enrichment_warning_reason_codes")
                    or []
                ),
                "feature_enrichment_post_market_close_candle_rejected_count": int(
                    report.get(
                        "feature_enrichment_post_market_close_candle_rejected_count"
                    )
                    or 0
                ),
                "raw_btc_candle_row_count": int(
                    report.get("raw_btc_candle_row_count") or 0
                ),
            }
        )
        return


def _recover_existing_exported_finalization(
    *,
    run_dir: Path,
    destination_root: Path,
    report: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = run_dir / "pending_round_finalization_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("existing exported finalization manifest is missing")
    manifest = _read_json(manifest_path)
    expected = {
        "run_id": run_dir.name,
        "finalization_status": "exported",
        "pending_resolution": False,
        "training_eligible": True,
    }
    for name, value in expected.items():
        if report.get(name) != value:
            raise ValueError(f"existing exported finalization report {name} mismatch")
    for name, value in (
        ("run_id", run_dir.name),
        ("finalization_status", "exported"),
        ("pending_resolution", False),
    ):
        if manifest.get(name) != value:
            raise ValueError(f"existing exported finalization manifest {name} mismatch")
    for payload_name, payload in (("report", report), ("manifest", manifest)):
        if not (
            payload.get("paper_only") is True
            and payload.get("capital_at_risk") is False
            and payload.get("polymarket_write_enabled") is False
            and payload.get("wallet_signing_enabled") is False
        ):
            raise ValueError(f"existing exported finalization {payload_name} safety mismatch")
    if int(report.get("raw_resolution_count") or 0) <= 0:
        raise ValueError("existing exported finalization resolution evidence is missing")
    if report.get("reject_reason_counts"):
        raise ValueError("existing exported finalization contains reject reasons")
    exported_value = str(report.get("exported_training_corpus_dir") or "")
    if exported_value != str(manifest.get("exported_training_corpus_dir") or ""):
        raise ValueError("existing exported finalization corpus path mismatch")
    exported_dir = Path(exported_value).expanduser().resolve()
    if not exported_dir.is_relative_to(destination_root.expanduser().resolve()):
        raise ValueError("existing exported finalization corpus is outside destination root")
    local_manifest_path = run_dir / "phase2_corpus" / "polymarket_corpus_manifest.json"
    exported_manifest_path = exported_dir / "polymarket_corpus_manifest.json"
    if not local_manifest_path.is_file() or not exported_manifest_path.is_file():
        raise ValueError("existing exported corpus manifest is missing")
    local_sha256 = _sha256_file(local_manifest_path)
    exported_sha256 = _sha256_file(exported_manifest_path)
    expected_sha256 = str(report.get("phase2_corpus_manifest_sha256") or "")
    if manifest.get("phase2_corpus_manifest_sha256") != expected_sha256:
        raise ValueError("existing exported finalization manifest hash lineage mismatch")
    if not expected_sha256 or local_sha256 != expected_sha256 or exported_sha256 != expected_sha256:
        raise ValueError("existing exported corpus manifest hash mismatch")
    return {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "finalization_status": "exported",
        "pending_resolution": False,
        "training_eligible": True,
        "exported_training_corpus_dir": str(exported_dir),
        "raw_resolution_count": int(report["raw_resolution_count"]),
        "reject_reason_counts": {},
        "recovered_from_existing_exported_report": True,
        "existing_finalization_report_sha256": _sha256_file(
            run_dir / "pending_round_finalization_report.json"
        ),
        "existing_finalization_manifest_sha256": _sha256_file(manifest_path),
        "exported_corpus_manifest_sha256": expected_sha256,
    }


def _outcome_blind_quality_control_snapshot(
    captures: list[dict[str, Any]],
    *,
    collector_contract: dict[str, Any],
    target_count: int,
) -> dict[str, Any]:
    """Count only causal capture-quality evidence; never inspect final outcomes."""

    ordered = sorted(
        captures,
        key=lambda row: (
            int(row.get("scheduled_round_start_ts") or 0),
            int(row.get("round_index") or 0),
            str(row.get("run_id") or ""),
        ),
    )
    valid: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    seen_boundaries: set[int] = set()
    for capture in ordered:
        audit = _capture_quality_audit(
            capture,
            collector_contract=collector_contract,
            finalization=None,
        )
        boundary = int(capture.get("scheduled_round_start_ts") or 0)
        reasons = list(audit["reason_codes"])
        if boundary in seen_boundaries:
            reasons.append("duplicate_scheduled_round_start")
        if boundary > 0:
            seen_boundaries.add(boundary)
        if reasons:
            exclusions.update(sorted(set(reasons)))
            continue
        valid.append(capture)
    selected = valid[:target_count]
    return {
        "outcome_blind_quality_valid_capture_count": len(valid),
        "outcome_blind_quality_valid_capture_run_ids": [
            str(row.get("run_id") or "") for row in selected
        ],
        "outcome_blind_quality_excluded_capture_count": len(ordered) - len(valid),
        "outcome_blind_quality_exclusion_reason_distribution": dict(
            sorted(exclusions.items())
        ),
        "quality_target_reached": len(valid) >= target_count,
    }


def _summary(
    batch_id: str,
    captures: list[dict[str, Any]],
    finalizations: list[dict[str, Any]],
    errors: list[dict[str, str]],
    *,
    quality_control_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    exported = [item for item in finalizations if item.get("finalization_status") == "exported"]
    pending = [
        item for item in finalizations if item.get("finalization_status") == "pending_resolution"
    ]
    pending_feature_enrichment = [
        item
        for item in finalizations
        if item.get("finalization_status") == "pending_feature_enrichment"
    ]
    market_identity_source_distribution: dict[str, int] = {}
    market_identity_fallback_reason_distribution: dict[str, int] = {}
    clob_revalidation_attempt_distribution: dict[str, int] = {}
    clob_revalidation_retry_reason_distribution: dict[str, int] = {}
    for capture in captures:
        for name, count in dict(
            capture.get(
                "provider_raw_market_identity_source_type_distribution"
            )
            or {}
        ).items():
            market_identity_source_distribution[str(name)] = (
                market_identity_source_distribution.get(str(name), 0)
                + int(count)
            )
        for name, count in dict(
            capture.get("market_identity_cache_fallback_reason_distribution")
            or {}
        ).items():
            market_identity_fallback_reason_distribution[str(name)] = (
                market_identity_fallback_reason_distribution.get(str(name), 0)
                + int(count)
            )
        for name, count in dict(
            capture.get(
                "market_identity_clob_revalidation_attempt_distribution"
            )
            or {}
        ).items():
            clob_revalidation_attempt_distribution[str(name)] = (
                clob_revalidation_attempt_distribution.get(str(name), 0)
                + int(count)
            )
        for name, count in dict(
            capture.get(
                "market_identity_clob_revalidation_retry_reason_distribution"
            )
            or {}
        ).items():
            clob_revalidation_retry_reason_distribution[str(name)] = (
                clob_revalidation_retry_reason_distribution.get(str(name), 0)
                + int(count)
            )
    summary = {
        "batch_id": batch_id,
        "paper_only": True,
        "capital_at_risk": False,
        "capture_count": len(captures),
        "capture_pending_resolution_count": sum(
            1 for item in captures if item.get("pending_resolution") is True
        ),
        "capture_pending_feature_enrichment_count": sum(
            1
            for item in captures
            if item.get("pending_feature_enrichment") is True
        ),
        "feature_enrichment_recovered_capture_count": sum(
            1
            for item in captures
            if item.get("feature_enrichment_recovered") is True
        ),
        "market_identity_source_type_distribution": dict(
            sorted(market_identity_source_distribution.items())
        ),
        "market_identity_cache_fallback_market_count": sum(
            int(item.get("market_identity_cache_fallback_market_count") or 0)
            for item in captures
        ),
        "market_identity_cache_fallback_reason_distribution": dict(
            sorted(market_identity_fallback_reason_distribution.items())
        ),
        "market_identity_cache_provenance_violation_count": sum(
            int(
                item.get(
                    "market_identity_cache_provenance_violation_count"
                )
                or 0
            )
            for item in captures
        ),
        "market_identity_clob_revalidation_passed_count": sum(
            int(
                item.get("market_identity_clob_revalidation_passed_count")
                or 0
            )
            for item in captures
        ),
        "market_identity_clob_revalidation_retry_succeeded_market_count": sum(
            int(
                item.get(
                    "market_identity_clob_revalidation_retry_succeeded_market_count"
                )
                or 0
            )
            for item in captures
        ),
        "market_identity_clob_revalidation_attempt_distribution": dict(
            sorted(clob_revalidation_attempt_distribution.items())
        ),
        "market_identity_clob_revalidation_retry_reason_distribution": dict(
            sorted(clob_revalidation_retry_reason_distribution.items())
        ),
        "market_identity_clob_revalidation_identity_relaxation_count": sum(
            int(
                item.get(
                    "market_identity_clob_revalidation_identity_relaxation_count"
                )
                or 0
            )
            for item in captures
        ),
        "provider_raw_orderbook_snapshot_count": sum(
            int(item.get("provider_raw_orderbook_snapshot_count") or 0) for item in captures
        ),
        "training_sampled_orderbook_row_count": sum(
            int(item.get("training_sampled_orderbook_row_count") or 0) for item in captures
        ),
        "raw_chainlink_price_row_count": sum(
            int(item.get("raw_chainlink_price_row_count") or 0) for item in captures
        ),
        "chainlink_covered_capture_count": sum(
            1 for item in captures if int(item.get("raw_chainlink_price_row_count") or 0) > 0
        ),
        "chainlink_fresh_capture_count": sum(
            1
            for item in captures
            if item.get("chainlink_rtds_price_stream_fresh") is True
        ),
        "chainlink_rtds_stale_reconnect_count": max(
            (
                int(item.get("chainlink_rtds_stale_reconnect_count") or 0)
                for item in captures
            ),
            default=0,
        ),
        "finalization_attempt_count": len(finalizations),
        "exported_round_count": len(exported),
        "pending_resolution_count": len(pending),
        "pending_feature_enrichment_count": len(
            pending_feature_enrichment
        ),
        "feature_enrichment_recovered_count": sum(
            1
            for item in finalizations
            if item.get("feature_enrichment_recovered") is True
        ),
        "feature_enrichment_post_market_close_candle_rejected_count": sum(
            int(
                item.get(
                    "feature_enrichment_post_market_close_candle_rejected_count"
                )
                or 0
            )
            for item in captures
        ),
        "error_count": len(errors),
        "captures": captures,
        "finalizations": finalizations,
        "errors": errors,
    }
    if quality_control_state is not None:
        summary.update(dict(quality_control_state))
    return summary


def _upsert_by_run_id(items: list[dict[str, Any]], item: dict[str, Any]) -> None:
    run_id = item["run_id"]
    for index, existing in enumerate(items):
        if existing.get("run_id") == run_id:
            items[index] = item
            return
    items.append(item)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _sleep_until_round_start_window(
    *,
    market_family: str,
    max_round_start_lag_seconds: float,
    previous_round_start_epoch_seconds: float | None = None,
) -> float:
    now_epoch_seconds = time.time()
    scheduled_round_start_epoch_seconds = _scheduled_round_start_epoch_seconds(
        market_family=market_family,
        max_round_start_lag_seconds=max_round_start_lag_seconds,
        now_epoch_seconds=now_epoch_seconds,
        previous_round_start_epoch_seconds=previous_round_start_epoch_seconds,
    )
    sleep_seconds = max(0.0, scheduled_round_start_epoch_seconds - now_epoch_seconds)
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)
    return scheduled_round_start_epoch_seconds


def _wait_until_scheduled_round_start(
    scheduled_round_start_epoch_seconds: float,
    *,
    now_fn: Callable[[], float] = time.time,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> float:
    """Prevent an early-woken worker from querying before its market boundary."""

    while True:
        now_epoch_seconds = now_fn()
        remaining_seconds = scheduled_round_start_epoch_seconds - now_epoch_seconds
        if remaining_seconds <= 0:
            return now_epoch_seconds
        sleep_fn(remaining_seconds)


def _scheduled_round_start_epoch_seconds(
    *,
    market_family: str,
    max_round_start_lag_seconds: float,
    now_epoch_seconds: float,
    previous_round_start_epoch_seconds: float | None,
) -> float:
    horizon_seconds = BTC_UPDOWN_MARKET_HORIZONS_MS[market_family] / 1000.0
    current_round_start = now_epoch_seconds - (now_epoch_seconds % horizon_seconds)
    elapsed = now_epoch_seconds - current_round_start
    if previous_round_start_epoch_seconds is None:
        return (
            current_round_start
            if elapsed <= max_round_start_lag_seconds
            else current_round_start + horizon_seconds
        )
    next_after_previous = previous_round_start_epoch_seconds + horizon_seconds
    if current_round_start > previous_round_start_epoch_seconds:
        current_or_next = (
            current_round_start
            if elapsed <= max_round_start_lag_seconds
            else current_round_start + horizon_seconds
        )
        return max(next_after_previous, current_or_next)
    return next_after_previous


def _round_start_alignment_sleep_seconds(
    *,
    market_family: str,
    max_round_start_lag_seconds: float,
    now_epoch_seconds: float,
) -> float:
    horizon_seconds = BTC_UPDOWN_MARKET_HORIZONS_MS[market_family] / 1000.0
    elapsed = now_epoch_seconds % horizon_seconds
    if elapsed <= max_round_start_lag_seconds:
        return 0.0
    return horizon_seconds - elapsed + 1.0


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    _write_json(temporary, payload)
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", default=f"async-round-collector-{_utc_stamp()}")
    parser.add_argument("--output-dir", default="examples/v8/operator_runs")
    parser.add_argument("--round-count", type=int, default=1)
    parser.add_argument(
        "--market-family",
        choices=tuple(BTC_UPDOWN_MARKET_HORIZONS_MS),
        default="btc_updown_5m",
    )
    parser.add_argument("--public-provider-timeout-seconds", type=float, default=330.0)
    parser.add_argument("--public-provider-http-timeout-seconds", type=float, default=15.0)
    parser.add_argument("--orderbook-snapshot-interval-seconds", type=float, default=1.0)
    parser.add_argument(
        "--orderbook-ws-initial-complete-book-timeout-seconds",
        type=float,
        default=15.0,
    )
    parser.add_argument(
        "--rest-orderbook-fallback-collection-seconds",
        type=float,
        default=330.0,
    )
    parser.add_argument("--settlement-poll-interval-seconds", type=float, default=15.0)
    parser.add_argument("--settlement-grace-seconds", type=float, default=0.0)
    parser.add_argument("--training-corpus-root", default=str(V8_TRAINING_CORPUS_ROOT))
    parser.add_argument("--clob-ws-url", default=DEFAULT_POLYMARKET_CLOB_WS_MARKET_URL)
    parser.add_argument("--max-round-start-lag-seconds", type=float, default=30.0)
    parser.add_argument("--chainlink-rtds-url", default=DEFAULT_POLYMARKET_RTDS_URL)
    parser.add_argument("--chainlink-rtds-warmup-seconds", type=float, default=5.0)
    parser.add_argument(
        "--chainlink-rtds-stale-reconnect-seconds",
        type=float,
        default=15.0,
    )
    parser.add_argument(
        "--market-identity-cache-path",
        default=None,
        help=(
            "Shared causal Gamma identity cache path; defaults to "
            "<output-dir>/gamma_market_identity_cache.json."
        ),
    )
    parser.add_argument(
        "--gamma-market-identity-prefetch-round-count",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--market-identity-cache-max-age-seconds",
        type=float,
        default=7_200.0,
    )
    parser.add_argument(
        "--clob-identity-revalidation-max-attempts",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--clob-identity-revalidation-retry-seconds",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--feature-enrichment-max-attempts",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--outcome-blind-quality-stop-target",
        type=int,
        default=None,
        help=(
            "Optional frozen quality-valid capture target. Requires the #190 "
            "collection freeze manifest and stops before the next round boundary."
        ),
    )
    parser.add_argument(
        "--outcome-blind-collection-only",
        action="store_true",
        help=(
            "Persist raw pending-round evidence without calling resolution, "
            "settlement finalization, or training-corpus export."
        ),
    )
    parser.add_argument(
        "--future-holdout-collection-freeze-manifest",
        default=None,
    )
    parser.add_argument(
        "--future-holdout-collection-freeze-manifest-sha256",
        default=None,
    )
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Only scan pending round captures and try settlement finalization.",
    )
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args(argv)
    if args.finalize_only:
        summary = run_polymarket_async_finalizer_cli(
            batch_id=args.batch_id,
            output_dir=args.output_dir,
            settlement_poll_interval_seconds=args.settlement_poll_interval_seconds,
            settlement_grace_seconds=args.settlement_grace_seconds,
            training_corpus_root=args.training_corpus_root,
            clob_ws_url=args.clob_ws_url,
            public_provider_http_timeout_seconds=(
                args.public_provider_http_timeout_seconds
            ),
            overwrite_existing=args.overwrite_existing,
        )
    else:
        summary = run_polymarket_async_round_collector_cli(
            batch_id=args.batch_id,
            output_dir=args.output_dir,
            round_count=args.round_count,
            market_family=args.market_family,
            public_provider_timeout_seconds=args.public_provider_timeout_seconds,
            public_provider_http_timeout_seconds=args.public_provider_http_timeout_seconds,
            orderbook_snapshot_interval_seconds=args.orderbook_snapshot_interval_seconds,
            orderbook_ws_initial_complete_book_timeout_seconds=(
                args.orderbook_ws_initial_complete_book_timeout_seconds
            ),
            rest_orderbook_fallback_collection_seconds=(
                args.rest_orderbook_fallback_collection_seconds
            ),
            settlement_poll_interval_seconds=args.settlement_poll_interval_seconds,
            settlement_grace_seconds=args.settlement_grace_seconds,
            training_corpus_root=args.training_corpus_root,
            clob_ws_url=args.clob_ws_url,
            max_round_start_lag_seconds=args.max_round_start_lag_seconds,
            chainlink_rtds_url=args.chainlink_rtds_url,
            chainlink_rtds_warmup_seconds=args.chainlink_rtds_warmup_seconds,
            chainlink_rtds_stale_reconnect_seconds=(
                args.chainlink_rtds_stale_reconnect_seconds
            ),
            market_identity_cache_path=args.market_identity_cache_path,
            gamma_market_identity_prefetch_round_count=(
                args.gamma_market_identity_prefetch_round_count
            ),
            market_identity_cache_max_age_seconds=(
                args.market_identity_cache_max_age_seconds
            ),
            clob_identity_revalidation_max_attempts=(
                args.clob_identity_revalidation_max_attempts
            ),
            clob_identity_revalidation_retry_seconds=(
                args.clob_identity_revalidation_retry_seconds
            ),
            feature_enrichment_max_attempts=(
                args.feature_enrichment_max_attempts
            ),
            outcome_blind_quality_stop_target=(
                args.outcome_blind_quality_stop_target
            ),
            outcome_blind_collection_only=args.outcome_blind_collection_only,
            future_holdout_collection_freeze_manifest=(
                args.future_holdout_collection_freeze_manifest
            ),
            future_holdout_collection_freeze_manifest_sha256=(
                args.future_holdout_collection_freeze_manifest_sha256
            ),
            overwrite_existing=args.overwrite_existing,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
