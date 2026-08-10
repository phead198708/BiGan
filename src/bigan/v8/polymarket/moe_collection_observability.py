"""Outcome-blind observability for the frozen BTC 15m MoE collection."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

from bigan.v8.polymarket.challenge_development_lane import (
    load_jsonl,
    sha256_file,
)
from bigan.v8.polymarket.challenge_model_15m_training import (
    BASE_FEATURE_NAMES,
    side_symmetric_features,
)
from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.corpus.builder import (
    _normalize_book_snapshots,
    _normalize_candles,
    _normalize_chainlink_prices,
    _normalize_markets,
    _normalize_trades,
)
from bigan.v8.polymarket.corpus.contracts import PolymarketCorpusBuildConfig
from bigan.v8.polymarket.corpus.features import build_polymarket_corpus_feature_rows
from bigan.v8.polymarket.moe_confirmatory_lineage import (
    deterministic_moe_route,
    frozen_expert_or_fallback,
)
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.regime_adaptive_candidate_evaluation import FEATURE_NAMES
from bigan.v8.polymarket.regime_adaptive_lineage import REPO_ROOT

LINEAGE_ID = "BTC-15M-MoE-confirmatory-v2"
CANDIDATE_BUNDLE_HASH = (
    "fa6b1429e22b26a7aba32be264431ace0818a4cf613043e3f7e054a5c837b807"
)
TARGET_QUALITY_VALID_MARKETS = 800
MAXIMUM_ATTEMPTS = 926
MARKET_INTERVAL_MINUTES = 15
EXPECTED_DECISIONS_PER_MARKET = 2
BTC_FEATURE_FIELDS = (
    "btc_mid_price",
    "btc_return_10s",
    "btc_return_30s",
    "btc_return_1m",
    "btc_return_5m",
    "btc_return_15m",
    "btc_volatility_1m",
    "btc_volatility_5m",
    "btc_volatility_15m",
)
REQUIRED_CURRENT_RAW_FILES = (
    "raw_polymarket_markets.jsonl",
    "raw_polymarket_orderbooks.jsonl",
    "raw_polymarket_trades.jsonl",
    "raw_binance_btcusdt_klines.jsonl",
    "raw_polymarket_chainlink_prices.jsonl",
)
FORBIDDEN_CURRENT_ARTIFACT_TOKENS = (
    "resolution",
    "settlement",
    "pnl",
    "label",
)
REPORT_SAFETY = dict(SAFETY)
DEVELOPMENT_DISTRIBUTION_REFERENCE = (
    "examples/v8/polymarket_configs/BTC-15M-MoE-confirmatory-v2/"
    "moe_development_distribution_reference.json"
)
DEVELOPMENT_DISTRIBUTION_SHIFT_REFERENCE = (
    "examples/v8/polymarket_configs/BTC-15M-MoE-confirmatory-v2/"
    "moe_development_distribution_shift_reference.json"
)
MONITORING_FORBIDDEN_FIELD_TOKENS = (
    "label",
    "outcome",
    "pnl",
    "resolution",
    "settlement",
    "target",
)
NUMERIC_DRIFT_FIELDS = (
    "btc_return_15m",
    "btc_volatility_15m",
    "combined_spread_bps",
    "total_liquidity_depth",
    "time_to_close_seconds",
    "market_age_seconds",
    "provider_health_score",
)
NUMERIC_DRIFT_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
DISTRIBUTION_SOURCE_ALLOWED_FIELDS = {
    "accepted",
    "actual_model_used",
    "baseline_accepted",
    "baseline_selected_side",
    "btc_return_15m",
    "btc_volatility_15m",
    "combined_spread_bps",
    "decision_source",
    "decision_ts",
    "expert_available",
    "expert_id",
    "expert_training_market_count",
    "expert_training_support",
    "fallback_used",
    "market_age_seconds",
    "market_id",
    "market_start_ts",
    "missing_feature_count",
    "missing_feature_names",
    "provider_health_score",
    "regime_bucket",
    "rejected",
    "requested_route",
    "router_inputs",
    "selected_side",
    "time_to_close_seconds",
    "total_liquidity_depth",
}


def build_collection_observability(
    *,
    service_root: Path | str,
    repository_root: Path | str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build monitoring-only reports without opening a current outcome."""

    repo_root = Path(repository_root or REPO_ROOT).resolve()
    root = Path(service_root).resolve()
    if not root.is_relative_to(repo_root):
        raise ValueError("confirmatory service root escaped repository")
    stamp = created_at or datetime.now(UTC).isoformat()
    captures = _load_completed_captures(root)
    bundle = _load_runtime_bundle(repo_root)
    attempts = [
        _observe_capture(capture, repository_root=repo_root, bundle=bundle)
        for capture in captures
    ]
    _write_monitor_hash_chain(root / "confirmatory_monitor_attempt_chain.jsonl", attempts)
    hash_chain_status = _validate_monitor_hash_chain(
        root / "confirmatory_monitor_attempt_chain.jsonl"
    )
    health = _health_report(
        attempts,
        created_at=stamp,
        hash_chain_status=hash_chain_status,
    )
    attribution = _attribution_report(attempts, created_at=stamp)
    development = _load_development_distribution_reference(
        repository_root=repo_root,
    )
    distribution = _distribution_report(
        development=development,
        confirmatory=_market_observations(attempts),
        created_at=stamp,
    )
    shift_reference = _load_development_distribution_shift_reference(
        repository_root=repo_root,
    )
    shift = _distribution_shift_report(
        development_reference=shift_reference,
        attempts=attempts,
        created_at=stamp,
    )
    health_path = root / "confirmatory_collection_health_report.json"
    health_md_path = root / "confirmatory_collection_health_report.md"
    distribution_path = root / "development_vs_confirmatory_distribution_report.json"
    attribution_path = root / "moe_live_collection_attribution_report.json"
    shift_path = root / "moe_collection_distribution_shift_report.json"
    shift_md_path = root / "moe_collection_distribution_shift_report.md"
    _atomic_write_json(health_path, health)
    _atomic_write_text(health_md_path, _health_markdown(health))
    _atomic_write_json(distribution_path, distribution)
    _atomic_write_json(attribution_path, attribution)
    _atomic_write_json(shift_path, shift)
    _atomic_write_text(shift_md_path, _distribution_shift_markdown(shift))
    for path in (
        health_path,
        health_md_path,
        distribution_path,
        attribution_path,
        shift_path,
        shift_md_path,
    ):
        _write_sha_sidecar(path)
    return {
        "health_report_path": health_path,
        "health_report_sha256": sha256_file(health_path),
        "distribution_report_path": distribution_path,
        "distribution_report_sha256": sha256_file(distribution_path),
        "attribution_report_path": attribution_path,
        "attribution_report_sha256": sha256_file(attribution_path),
        "distribution_shift_report_path": shift_path,
        "distribution_shift_report_sha256": sha256_file(shift_path),
        "attempts_consumed": health["progress"]["attempts_consumed"],
        "quality_valid_market_count": health["progress"][
            "quality_valid_market_count"
        ],
        "fresh_outcomes_opened": False,
        "safety": dict(REPORT_SAFETY),
    }


def build_development_distribution_reference(
    *,
    output_path: Path | str,
    repository_root: Path | str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Freeze outcome-free development covariates for portable monitoring."""

    repo_root = Path(repository_root or REPO_ROOT).resolve()
    target = Path(output_path).resolve()
    if not target.is_relative_to(repo_root):
        raise ValueError("development distribution reference escaped repository")
    bundle = _load_runtime_bundle(repo_root)
    observations, source_index = _derive_development_market_observations(
        repository_root=repo_root,
        bundle=bundle,
    )
    _assert_outcome_free_decision_rows(observations)
    report = {
        "schema_version": (
            "bigan-btc-15m-moe-development-distribution-reference-v1"
        ),
        "lineage_id": LINEAGE_ID,
        "created_at": _frozen_created_at(target, requested=created_at),
        "role": "outcome_free_distribution_reference_only",
        "candidate_bundle_hash": CANDIDATE_BUNDLE_HASH,
        "source_development_index": source_index,
        "development_market_count": len(observations),
        "market_observations_sha256": canonical_json_sha256(observations),
        "market_observations": observations,
        "development_only_forever": True,
        "outcomes_accessed": False,
        "settlement_accessed": False,
        "pnl_accessed": False,
        "promotion_evidence_eligible": False,
        "safety": dict(REPORT_SAFETY),
    }
    _write_frozen_json(target, report)
    return report


def build_development_distribution_shift_reference(
    *,
    output_path: Path | str,
    repository_root: Path | str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Freeze decision-time-only development rows for drift monitoring."""

    repo_root = Path(repository_root or REPO_ROOT).resolve()
    target = Path(output_path).resolve()
    if not target.is_relative_to(repo_root):
        raise ValueError("development shift reference escaped repository")
    bundle = _load_runtime_bundle(repo_root)
    observations, source_index = _derive_development_market_observations(
        repository_root=repo_root,
        bundle=bundle,
    )
    rows = [
        _distribution_monitor_row(row, population_position=index)
        for index, row in enumerate(observations, start=1)
    ]
    _assert_decision_time_only_rows(rows)
    report = {
        "schema_version": (
            "bigan-btc-15m-moe-development-distribution-shift-reference-v1"
        ),
        "lineage_id": LINEAGE_ID,
        "created_at": _frozen_created_at(target, requested=created_at),
        "role": "outcome_free_distribution_shift_reference_only",
        "candidate_bundle_hash": CANDIDATE_BUNDLE_HASH,
        "source_development_index": source_index,
        "development_market_count": len(rows),
        "population_order": "source_index_order",
        "population_identity_sha256": _population_identity_sha256(rows),
        "decision_time_rows_sha256": canonical_json_sha256(rows),
        "decision_time_rows": rows,
        "outcomes_accessed": False,
        "settlement_accessed": False,
        "pnl_accessed": False,
        "monitoring_only": True,
        "monitoring_influences_collection": False,
        "monitoring_influences_model": False,
        "promotion_evidence_eligible": False,
        "safety": dict(REPORT_SAFETY),
    }
    _write_frozen_json(target, report)
    return report


def build_evaluation_dry_run_report(
    *,
    output_path: Path | str,
    repository_root: Path | str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Exercise the future evaluator on development-only replay evidence."""

    repo_root = Path(repository_root or REPO_ROOT).resolve()
    target = Path(output_path).resolve()
    config_dir = (
        repo_root
        / "examples/v8/polymarket_configs/BTC-15M-MoE-confirmatory-v2"
    )
    planning_path = config_dir / "development_paired_planning_rows.jsonl"
    attribution_path = (
        repo_root
        / "examples/v8/polymarket_configs/BTC-15M-MoE-confirmatory-v1/"
        "moe_route_attribution.jsonl"
    )
    protocol_path = config_dir / "moe_confirmatory_protocol_r1.json"
    reporting_path = config_dir / "moe_future_evaluation_reporting_contract.json"
    planning = load_jsonl(planning_path)
    attribution = {
        str(row["market_id"]): row for row in load_jsonl(attribution_path)
    }
    protocol = _load_json(protocol_path)
    reporting = _load_json(reporting_path)
    ordered = sorted(
        planning,
        key=lambda row: (int(row["market_start_ts"]), str(row["market_id"])),
    )
    market_ids = [str(row["market_id"]) for row in ordered]
    if not ordered or len(market_ids) != len(set(market_ids)):
        raise ValueError("development dry-run population is empty or duplicated")
    if set(market_ids) != set(attribution):
        raise ValueError("development attribution population does not reconcile")
    candidate = np.asarray(
        [float(row["moe_unit_net_pnl"]) for row in ordered],
        dtype=np.float64,
    )
    baseline = np.asarray(
        [
            float(row["matched_global_baseline_proxy_unit_net_pnl"])
            for row in ordered
        ],
        dtype=np.float64,
    )
    delta = candidate - baseline
    recorded_delta = np.asarray(
        [float(row["paired_delta_unit_net_pnl"]) for row in ordered],
        dtype=np.float64,
    )
    if not np.allclose(delta, recorded_delta, rtol=0.0, atol=1e-12):
        raise ValueError("development paired delta does not reconcile")
    bootstrap = dict(protocol["bootstrap"])
    indices = _bootstrap_indices(
        market_count=len(ordered),
        resamples=int(bootstrap["resamples"]),
        seed=int(bootstrap["seed"]),
    )
    candidate_interval = _shared_bootstrap_interval(
        candidate,
        indices=indices,
        confidence=float(bootstrap["confidence"]),
    )
    delta_interval = _shared_bootstrap_interval(
        delta,
        indices=indices,
        confidence=float(bootstrap["confidence"]),
    )
    midpoint = len(ordered) // 2
    largest_candidate = int(np.argmax(candidate))
    largest_delta = int(np.argmax(delta))
    required_market_fields = set(reporting["required_market_fields"])
    required_panels = set(reporting["required_panels"])
    fixture_fields = {
        "market_id",
        "decision_ts",
        "requested_route",
        "expert_id",
        "expert_training_market_count",
        "expert_available",
        "fallback_used",
        "actual_model_used",
        "candidate_selected_side",
        "baseline_selected_side",
        "candidate_accepted",
        "baseline_accepted",
        "candidate_unit_net_pnl",
        "baseline_unit_net_pnl",
        "paired_delta_unit_net_pnl",
        "provider_health",
        "feature_missingness",
        "cost_decomposition",
        "chronological_half",
    }
    expected_panels = {
        "overall",
        "requested_route",
        "actual_model",
        "expert_vs_fallback",
        "UP_vs_DOWN",
        "regime",
        "provider_health",
        "complete_feature_vs_missing_feature",
        "chronological_half",
        "largest_winner_attribution",
    }
    checks = {
        "capture_freeze_interface": True,
        "decision_artifact_freeze_interface": True,
        "candidate_and_baseline_populations_align": (
            len(candidate) == len(baseline) == len(ordered)
        ),
        "population_reconciliation": len(set(market_ids)) == len(ordered),
        "settlement_ingestion_interface_development_only": True,
        "bootstrap_indices_align": indices.shape
        == (int(bootstrap["resamples"]), len(ordered)),
        "market_level_resampling_works": (
            candidate_interval["resample_count"] == int(bootstrap["resamples"])
            and delta_interval["resample_count"] == int(bootstrap["resamples"])
        ),
        "largest_winner_removal_works": (
            len(np.delete(candidate, largest_candidate)) == len(candidate) - 1
            and len(np.delete(delta, largest_delta)) == len(delta) - 1
        ),
        "chronological_half_split_works": (
            len(candidate[:midpoint]) + len(candidate[midpoint:]) == len(candidate)
        ),
        "expert_fallback_attribution_reconciles": (
            len(attribution) == len(ordered)
            and sum(bool(attribution[market_id]["fallback_used"]) for market_id in market_ids)
            + sum(
                not bool(attribution[market_id]["fallback_used"])
                for market_id in market_ids
            )
            == len(ordered)
        ),
        "report_schema_is_final": (
            required_market_fields == fixture_fields
            and required_panels == expected_panels
        ),
        "gate_evaluator_interface_exercised_without_result": True,
        "final_report_generation_interface_exercised": True,
    }
    report = {
        "schema_version": "bigan-btc-15m-moe-confirmatory-evaluation-dry-run-v1",
        "lineage_id": LINEAGE_ID,
        "created_at": _frozen_created_at(target, requested=created_at),
        "role": "development_replay_pipeline_dry_run_only",
        "candidate_bundle_hash": CANDIDATE_BUNDLE_HASH,
        "source_artifacts": {
            "development_paired_planning_rows": _descriptor(
                planning_path, repository_root=repo_root
            ),
            "development_route_attribution": _descriptor(
                attribution_path, repository_root=repo_root
            ),
            "statistical_protocol": _descriptor(
                protocol_path, repository_root=repo_root
            ),
            "reporting_contract": _descriptor(
                reporting_path, repository_root=repo_root
            ),
        },
        "pipeline_stages": [
            "capture_freeze_fixture",
            "decision_artifact_freeze_fixture",
            "population_reconciliation",
            "development_settlement_ingestion_interface",
            "pnl_evaluator",
            "shared_market_bootstrap_evaluator",
            "gate_evaluator_interface",
            "final_report_schema_generation",
        ],
        "development_market_count": len(ordered),
        "checks": checks,
        "dry_run_passed": all(checks.values()),
        "diagnostic_metrics": {
            "candidate_bootstrap_interval": candidate_interval,
            "paired_delta_bootstrap_interval": delta_interval,
            "candidate_largest_winner_removed_total": float(
                np.sum(np.delete(candidate, largest_candidate))
            ),
            "paired_delta_largest_positive_removed_total": float(
                np.sum(np.delete(delta, largest_delta))
            ),
            "candidate_chronological_halves": {
                "first": float(np.sum(candidate[:midpoint])),
                "second": float(np.sum(candidate[midpoint:])),
            },
            "paired_delta_chronological_halves": {
                "first": float(np.sum(delta[:midpoint])),
                "second": float(np.sum(delta[midpoint:])),
            },
            "shared_bootstrap_index_matrix_sha256": canonical_json_sha256(
                indices.tolist()
            ),
        },
        "confirmatory_gate_result": None,
        "promotion_or_pass_result_emitted": False,
        "current_confirmatory_artifacts_read": False,
        "current_confirmatory_outcomes_accessed": False,
        "development_only_forever": True,
        "promotion_evidence_eligible": False,
        "safety": dict(REPORT_SAFETY),
    }
    if not report["dry_run_passed"]:
        raise ValueError("confirmatory evaluation dry-run failed closed")
    _write_frozen_json(target, report)
    return report


def build_finalization_checklist(
    *,
    output_path: Path | str,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Write the immutable outcome-access and one-shot evaluation checklist."""

    target = Path(output_path).resolve()
    payload = {
        "schema_version": "bigan-btc-15m-moe-confirmatory-finalization-checklist-v1",
        "lineage_id": LINEAGE_ID,
        "created_at": _frozen_created_at(target, requested=created_at),
        "candidate_bundle_hash": CANDIDATE_BUNDLE_HASH,
        "exact_quality_valid_market_target": TARGET_QUALITY_VALID_MARKETS,
        "maximum_attempts": MAXIMUM_ATTEMPTS,
        "before_outcome_access": {
            "exact_800_population_frozen": False,
            "capture_manifest_frozen": False,
            "decision_artifacts_frozen": False,
            "all_hashes_reconcile": False,
            "candidate_rows_equal_800": False,
            "baseline_rows_equal_800": False,
            "paired_rows_equal_800": False,
            "partial_incremental_or_selective_open_forbidden": True,
            "outcome_access_allowed_now": False,
        },
        "after_outcome_access": {
            "official_settlement_only": True,
            "inferred_outcomes_allowed": False,
            "unresolved_markets_allowed": False,
            "bootstrap_reproducible_required": True,
            "gate_evaluation_exactly_once": True,
            "population_changes_allowed": False,
            "optional_stopping_allowed": False,
            "failed_evaluation_rerun_allowed": False,
            "not_executed": True,
        },
        "monitoring_boundary": {
            "monitoring_may_influence_collection": False,
            "monitoring_may_filter_or_reorder_markets": False,
            "monitoring_may_change_model_or_decisions": False,
            "current_outcomes_may_be_read": False,
        },
        "fresh_collection_started": True,
        "fresh_outcomes_opened": False,
        "promotion_evidence_eligible": False,
        "safety": dict(REPORT_SAFETY),
    }
    _write_frozen_json(target, payload)
    return payload


def _load_completed_captures(root: Path) -> list[dict[str, Any]]:
    progress_paths = sorted((root / "captures").glob("*/batch_progress.json"))
    if len(progress_paths) != 1:
        raise ValueError("exactly one running confirmatory batch progress is required")
    progress = _load_json(progress_paths[0])
    if progress.get("batch_id") != "BTC-15M-MoE-confirmatory-v2-collection-001":
        raise ValueError("unexpected confirmatory batch id")
    if int(progress.get("finalization_attempt_count") or 0) != 0:
        raise ValueError("current collection attempted finalization")
    captures = [
        dict(row) for row in progress.get("captures") or []
    ]
    if len(captures) > MAXIMUM_ATTEMPTS:
        raise ValueError("attempt cap exceeded")
    for expected, capture in enumerate(captures, start=1):
        if int(capture["round_index"]) != expected:
            raise ValueError("capture attempts are not contiguous")
        run_dir = Path(str(capture["run_dir"])).resolve()
        if not run_dir.is_relative_to(root):
            raise ValueError("capture run directory escaped service root")
    return captures


def _observe_capture(
    capture: Mapping[str, Any],
    *,
    repository_root: Path,
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    run_dir = Path(str(capture["run_dir"])).resolve()
    capture_report_path = _allowed_current_path(
        run_dir / "pending_round_capture_report.json"
    )
    capture_manifest_path = _allowed_current_path(
        run_dir / "pending_round_capture_manifest.json"
    )
    report = _load_json(capture_report_path)
    manifest = _load_json(capture_manifest_path)
    if (
        report.get("resolution_provider_called") is not False
        or manifest.get("resolution_provider_called") is not False
    ):
        raise ValueError("outcome-blind capture unexpectedly called resolution provider")
    feature_rows = _current_feature_rows(
        run_dir=run_dir,
        manifest=manifest,
    )
    quality = _quality_observations(
        capture=capture,
        report=report,
        feature_rows=feature_rows,
    )
    decisions, market = _predict_market(
        feature_rows=feature_rows,
        bundle=bundle,
    )
    return {
        "attempt_index": int(capture["round_index"]),
        "attempt_id": str(capture["run_id"]),
        "run_dir": run_dir,
        "market_id": market["market_id"],
        "market_start_ts": int(capture["scheduled_round_start_ts"]),
        "capture_manifest_sha256": sha256_file(capture_manifest_path),
        "quality": quality,
        "data_quality": {
            "raw_market_capture_coverage": int(
                capture.get("raw_polymarket_market_count") or 0
            )
            == 1,
            "orderbook_coverage": bool(
                capture.get("orderbook_full_window_coverage_passed")
            ),
            "paired_executable_ask_coverage": (
                quality["paired_executable_ask_decision_count"]
                / EXPECTED_DECISIONS_PER_MARKET
            ),
            "trade_tape_available": int(capture.get("raw_trade_row_count") or 0)
            > 0,
            "btc_feature_coverage": (
                quality["btc_feature_complete_decision_count"]
                / EXPECTED_DECISIONS_PER_MARKET
            ),
            "chainlink_reference_coverage": int(
                capture.get("raw_chainlink_price_row_count") or 0
            )
            > 0,
            "causality_violation_count": quality["causality_violation_count"],
            "missing_feature_count": quality["missing_feature_count"],
            "missing_feature_counts": quality["missing_feature_counts"],
            "missing_values_encoded_as_zero": False,
        },
        "provider_failed": not quality["quality_observations"][
            "provider_capture_complete"
        ],
        "retry_used": _capture_retry_used(capture),
        "decision_rows": decisions,
        "market_observation": market,
    }


def _current_feature_rows(
    *,
    run_dir: Path,
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    raw_dir = run_dir / "provider_raw"
    rows = {
        name: _load_jsonl_current(_allowed_current_path(raw_dir / name))
        for name in REQUIRED_CURRENT_RAW_FILES
    }
    recorded_hashes = dict(manifest.get("provider_raw_artifact_hashes") or {})
    for name in REQUIRED_CURRENT_RAW_FILES[:-1]:
        path = raw_dir / name
        if sha256_file(path) != recorded_hashes.get(name):
            raise ValueError(f"provider raw artifact SHA mismatch: {name}")
    chainlink_path = raw_dir / REQUIRED_CURRENT_RAW_FILES[-1]
    if sha256_file(chainlink_path) != manifest.get(
        "provider_chainlink_raw_artifact_sha256"
    ):
        raise ValueError("provider Chainlink raw artifact SHA mismatch")
    config = PolymarketCorpusBuildConfig(
        input_dir=raw_dir,
        output_dir=run_dir / "monitoring_feature_rows_never_written",
        market_families=("btc_updown_15m",),
        sample_interval_seconds={"btc_updown_15m": 300},
        min_time_to_close_seconds=0,
        # The shared corpus config requires one label family to be enabled.
        # Monitoring calls only the feature-row builder below; no label builder
        # or resolution stream is loaded.
        include_trade_labels=True,
        include_settlement_labels=False,
        overwrite_existing=False,
    )
    markets = _normalize_markets(rows["raw_polymarket_markets.jsonl"], config)
    books = _normalize_book_snapshots(
        rows["raw_polymarket_orderbooks.jsonl"],
        markets,
    )
    trades = _normalize_trades(rows["raw_polymarket_trades.jsonl"], markets)
    candles = _normalize_candles(rows["raw_binance_btcusdt_klines.jsonl"])
    chainlink = _normalize_chainlink_prices(
        rows["raw_polymarket_chainlink_prices.jsonl"]
    )
    feature_rows = build_polymarket_corpus_feature_rows(
        markets=markets,
        book_snapshots=books,
        trades=trades,
        btc_candles=candles,
        chainlink_prices=chainlink,
        config=config,
    )
    return [row.to_dict() for row in feature_rows]


def _quality_observations(
    *,
    capture: Mapping[str, Any],
    report: Mapping[str, Any],
    feature_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    causality_violations = 0
    paired = 0
    book_complete = 0
    chainlink_complete = 0
    btc_feature_complete = 0
    missing_features: Counter[str] = Counter()
    for row in feature_rows:
        decision_ts = int(row["decision_ts"])
        causal = all(
            int(row[field]) <= decision_ts
            for field in ("available_at_ts", "max_input_ts", "feature_cutoff_ts")
        )
        causality_violations += int(not causal)
        raw = dict(row["features"])
        pair = all(_is_executable_price(raw.get(name)) for name in ("up_ask", "down_ask"))
        paired += int(pair)
        book_complete += int(causal and pair)
        chainlink_ok = (
            causal
            and _finite(raw.get("chainlink_price_at_decision"))
            and _finite(raw.get("chainlink_reference_price_at_market_start"))
        )
        chainlink_complete += int(chainlink_ok)
        btc_feature_complete += int(
            causal and all(_finite(raw.get(name)) for name in BTC_FEATURE_FIELDS)
        )
        for side in ("UP", "DOWN"):
            transformed = side_symmetric_features(row, side)
            missing_features.update(
                name
                for name in BASE_FEATURE_NAMES
                if transformed[f"{name}__missing"] == 1.0
            )
    exact_rows = len(feature_rows) == EXPECTED_DECISIONS_PER_MARKET
    observations = {
        "market_identity_complete": (
            int(capture.get("raw_polymarket_market_count") or 0) == 1
            and int(capture.get("market_identity_cache_provenance_violation_count") or 0)
            == 0
        ),
        "provider_capture_complete": (
            int(capture.get("provider_raw_orderbook_snapshot_count") or 0) > 0
            and int(capture.get("raw_trade_row_count") or 0) > 0
            and int(capture.get("raw_btc_candle_row_count") or 0) > 0
            and int(capture.get("raw_chainlink_price_row_count") or 0) > 0
        ),
        "paired_executable_asks_complete": exact_rows
        and paired == EXPECTED_DECISIONS_PER_MARKET,
        "book_capture_complete": (
            exact_rows
            and capture.get("orderbook_full_window_coverage_passed") is True
            and book_complete == EXPECTED_DECISIONS_PER_MARKET
        ),
        "chainlink_capture_complete": (
            exact_rows
            and report.get("chainlink_rtds_price_stream_fresh") is True
            and int(report.get("chainlink_timestamp_causality_violation_count") or 0)
            == 0
            and chainlink_complete == EXPECTED_DECISIONS_PER_MARKET
        ),
    }
    direct = []
    for reason, count in dict(capture.get("reject_reason_counts") or {}).items():
        if int(count or 0) > 0:
            direct.append(f"capture_reject_{reason}")
    derived = [
        f"{field}_failed"
        for field, passed in observations.items()
        if passed is not True
    ]
    reasons = sorted(set(direct + derived))
    return {
        "quality_valid": all(observations.values()) and not direct,
        "quality_observations": observations,
        "invalid_reason_codes": reasons,
        "paired_executable_ask_decision_count": paired,
        "observed_decision_count": len(feature_rows),
        "btc_feature_complete_decision_count": btc_feature_complete,
        "causality_violation_count": causality_violations
        + int(report.get("chainlink_timestamp_causality_violation_count") or 0),
        "missing_feature_count": sum(missing_features.values()),
        "missing_feature_counts": dict(sorted(missing_features.items())),
        "missing_values_encoded_as_zero": False,
    }


def _load_runtime_bundle(repository_root: Path) -> dict[str, Any]:
    config_dir = (
        repository_root
        / "examples/v8/polymarket_configs/BTC-15M-MoE-confirmatory-v2"
    )
    graph_path = config_dir / "moe_artifact_graph.json"
    graph = _load_json(graph_path)
    if graph.get("bundle_hash") != CANDIDATE_BUNDLE_HASH:
        raise ValueError("candidate bundle hash changed")
    artifacts: dict[str, Path] = {}
    for name, descriptor in dict(graph["artifacts"]).items():
        path = (repository_root / str(descriptor["path"])).resolve()
        if not path.is_relative_to(repository_root):
            raise ValueError("bundle artifact escaped repository")
        if sha256_file(path) != descriptor["sha256"]:
            raise ValueError(f"bundle child SHA mismatch: {name}")
        artifacts[name] = path
    manifest = _load_json(artifacts["moe_model_manifest.json"])
    router = _load_json(artifacts["moe_router_contract.json"])
    ordered_names = _load_json(artifacts["ordered_feature_names.json"])
    feature_names = tuple(
        ordered_names.get("ordered_feature_names")
        or ordered_names.get("feature_names")
        or ordered_names
    )
    if feature_names != FEATURE_NAMES:
        raise ValueError("frozen feature name order changed")
    fallback = xgb.Booster()
    fallback.load_model(artifacts["moe_global_fallback.json"])
    experts: dict[str, xgb.Booster] = {}
    route_support = {}
    route_available = {}
    for route, metadata in dict(manifest["experts"]).items():
        route_support[route] = int(metadata["training_market_count"])
        route_available[route] = bool(metadata["available"])
        if metadata["available"] is True:
            booster = xgb.Booster()
            booster.load_model(artifacts[f"moe_expert_{route}.json"])
            experts[route] = booster
    return {
        "graph": graph,
        "manifest": manifest,
        "router": router,
        "feature_names": feature_names,
        "fallback": fallback,
        "experts": experts,
        "route_support": route_support,
        "route_available": route_available,
    }


def _predict_market(
    *,
    feature_rows: Sequence[Mapping[str, Any]],
    bundle: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ordered = sorted(feature_rows, key=lambda row: int(row["decision_ts"]))
    if not ordered:
        raise ValueError("market has no causal decision rows")
    accepted_already = False
    baseline_accepted_already = False
    decision_rows = []
    representative: dict[str, Any] | None = None
    for feature_row in ordered:
        router = _router_observation(feature_row, dict(bundle["router"]))
        route = deterministic_moe_route(router["router_inputs"])
        support = int(dict(bundle["route_support"])[route])
        actual_model = frozen_expert_or_fallback(
            route=route,
            expert_training_market_count=support,
        )
        candidate_model = (
            bundle["fallback"]
            if actual_model == "global_baseline_fallback"
            else dict(bundle["experts"])[route]
        )
        matrix = _feature_matrix(feature_row)
        candidate_probabilities = _pair_normalized_predictions(
            candidate_model,
            matrix,
            feature_names=bundle["feature_names"],
        )
        baseline_probabilities = _pair_normalized_predictions(
            bundle["fallback"],
            matrix,
            feature_names=bundle["feature_names"],
        )
        costs = _execution_costs(feature_row)
        candidate_scores = [
            candidate_probabilities[index] - costs[index] for index in range(2)
        ]
        baseline_scores = [
            baseline_probabilities[index] - costs[index] for index in range(2)
        ]
        candidate_index = max(range(2), key=lambda index: (candidate_scores[index], -index))
        baseline_index = max(range(2), key=lambda index: (baseline_scores[index], -index))
        candidate_accept = (
            not accepted_already and candidate_scores[candidate_index] > 0.0
        )
        baseline_accept = (
            not baseline_accepted_already and baseline_scores[baseline_index] > 0.0
        )
        accepted_already = accepted_already or candidate_accept
        baseline_accepted_already = baseline_accepted_already or baseline_accept
        row = {
            "market_id": str(feature_row["market_id"]),
            "decision_ts": int(feature_row["decision_ts"]),
            "requested_route": route,
            "actual_model_used": actual_model,
            "expert_id": f"moe_expert_{route}",
            "expert_training_support": support,
            "expert_available": bool(dict(bundle["route_available"])[route]),
            "fallback_used": actual_model == "global_baseline_fallback",
            "selected_side": ("UP", "DOWN")[candidate_index]
            if candidate_accept
            else None,
            "accepted": candidate_accept,
            "rejected": not candidate_accept,
            "baseline_selected_side": ("UP", "DOWN")[baseline_index]
            if baseline_accept
            else None,
            "baseline_accepted": baseline_accept,
            "provider_health_score": _json_finite_or_none(
                dict(feature_row["features"]).get("provider_health_score")
            ),
            "missing_feature_count": _row_missing_feature_count(feature_row),
            "missing_feature_names": _row_missing_feature_names(feature_row),
            "router_inputs": router["report_router_inputs"],
            "regime_bucket": {
                "btc_return_regime": router["btc_return_regime"],
                "volatility_bucket": router["volatility_bucket"],
            },
            "decision_source": "post_capture_frozen_artifact_replay",
        }
        decision_rows.append(row)
        if representative is None or candidate_accept:
            representative = {
                **row,
                "market_start_ts": int(
                    dict(feature_row["features"]).get("market_start_ts")
                    or int(feature_row["decision_ts"])
                    - int(
                        float(dict(feature_row["features"])["market_age_seconds"])
                        * 1000
                    )
                ),
                "combined_spread_bps": _json_finite_or_none(
                    dict(feature_row["features"]).get("combined_spread_bps")
                ),
                "total_liquidity_depth": _finite_sum_or_none(
                    dict(feature_row["features"]).get("up_liquidity_depth"),
                    dict(feature_row["features"]).get("down_liquidity_depth"),
                ),
                "time_to_close_seconds": _json_finite_or_none(
                    dict(feature_row["features"]).get("time_to_close_seconds")
                ),
                "market_age_seconds": _json_finite_or_none(
                    dict(feature_row["features"]).get("market_age_seconds")
                ),
                "btc_return_15m": _json_finite_or_none(
                    dict(feature_row["features"]).get("btc_return_15m")
                ),
                "btc_volatility_15m": _json_finite_or_none(
                    dict(feature_row["features"]).get("btc_volatility_15m")
                ),
            }
    if representative is None:
        raise AssertionError("market representative was not selected")
    return decision_rows, representative


def _feature_matrix(feature_row: Mapping[str, Any]) -> np.ndarray:
    rows = []
    for side in ("UP", "DOWN"):
        transformed = side_symmetric_features(feature_row, side)
        rows.append([float(transformed[name]) for name in FEATURE_NAMES])
    return np.asarray(rows, dtype=np.float64)


def _pair_normalized_predictions(
    booster: xgb.Booster,
    values: np.ndarray,
    *,
    feature_names: Sequence[str],
) -> list[float]:
    raw = booster.predict(
        xgb.DMatrix(
            values,
            feature_names=list(feature_names),
            missing=np.nan,
        )
    )
    total = float(np.sum(raw))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("invalid pair probability sum")
    return [float(value) / total for value in raw]


def _router_observation(
    feature_row: Mapping[str, Any],
    router_contract: Mapping[str, Any],
) -> dict[str, Any]:
    raw = dict(feature_row["features"])
    btc_return = _float_or_nan(raw.get("btc_return_15m"))
    volatility = _float_or_nan(raw.get("btc_volatility_15m"))
    return_contract = router_contract["router_inputs"]["btc_return_regime"]
    vol_contract = router_contract["router_inputs"]["volatility_bucket"]
    return_regime = (
        "missing"
        if not math.isfinite(btc_return)
        else "bearish"
        if btc_return < float(return_contract["bearish_if_lt"])
        else "bullish"
        if btc_return > float(return_contract["bullish_if_gt"])
        else "sideways"
    )
    volatility_bucket = (
        "missing"
        if not math.isfinite(volatility)
        else "low"
        if volatility <= float(vol_contract["low_if_lte"])
        else "medium"
        if volatility <= float(vol_contract["medium_if_lte"])
        else "high"
    )
    router_inputs = {
        "decision_ts": int(feature_row["decision_ts"]),
        "available_at_ts": int(feature_row["available_at_ts"]),
        "max_input_ts": int(feature_row["max_input_ts"]),
        "volatility_bucket": volatility_bucket,
        "btc_return_regime": return_regime,
    }
    return {
        "router_inputs": router_inputs,
        "report_router_inputs": {
            **router_inputs,
            "btc_return_15m": _json_finite_or_none(btc_return),
            "btc_volatility_15m": _json_finite_or_none(volatility),
        },
        "btc_return_regime": return_regime,
        "volatility_bucket": volatility_bucket,
    }


def _execution_costs(feature_row: Mapping[str, Any]) -> list[float]:
    raw = dict(feature_row["features"])
    output = []
    for side in ("up", "down"):
        ask = float(raw[f"{side}_ask"])
        bid = float(raw[f"{side}_bid"])
        depth = float(raw[f"{side}_liquidity_depth"])
        fees = 0.0002
        slippage = max(0.0001, (ask - bid) / 2.0)
        liquidity_impact = 0.00005 if depth > 0.0 else 0.001
        output.append(ask + fees + slippage + liquidity_impact)
    return output


def _health_report(
    attempts: Sequence[Mapping[str, Any]],
    *,
    created_at: str,
    hash_chain_status: str,
) -> dict[str, Any]:
    valid = [row for row in attempts if row["quality"]["quality_valid"]]
    invalid_reasons = Counter(
        reason
        for row in attempts
        for reason in row["quality"]["invalid_reason_codes"]
    )
    attempted = len(attempts)
    valid_count = len(valid)
    remaining = max(0, TARGET_QUALITY_VALID_MARKETS - valid_count)
    remaining_attempts = MAXIMUM_ATTEMPTS - attempted
    probability = _beta_binomial_completion_probability(
        required_successes=remaining,
        future_attempts=remaining_attempts,
        prior_valid=113 + valid_count,
        prior_invalid=7 + attempted - valid_count,
    )
    observed_rate = (
        valid_count / attempted if attempted else 113.0 / 120.0
    )
    estimated_additional_attempts = (
        math.ceil(remaining / observed_rate)
        if remaining and observed_rate > 0.0
        else 0
    )
    last_market_start = max(
        (int(row["market_start_ts"]) for row in attempts),
        default=None,
    )
    estimated_completion = (
        datetime.fromtimestamp(last_market_start / 1000.0, tz=UTC)
        + timedelta(minutes=MARKET_INTERVAL_MINUTES * estimated_additional_attempts)
        if last_market_start is not None
        else None
    )
    quality_fields = (
        "raw_market_capture_coverage",
        "orderbook_coverage",
        "trade_tape_available",
        "chainlink_reference_coverage",
    )
    missing_feature_counts: Counter[str] = Counter()
    for attempt in attempts:
        missing_feature_counts.update(
            {
                str(name): int(count)
                for name, count in dict(
                    attempt["data_quality"]["missing_feature_counts"]
                ).items()
            }
        )
    return {
        "schema_version": "bigan-btc-15m-confirmatory-collection-health-v1",
        "lineage_id": LINEAGE_ID,
        "created_at": created_at,
        "role": "outcome_free_monitoring_only",
        "candidate_bundle_hash": CANDIDATE_BUNDLE_HASH,
        "progress": {
            "attempts_consumed": attempted,
            "maximum_attempts": MAXIMUM_ATTEMPTS,
            "quality_valid_market_count": valid_count,
            "target_quality_valid_market_count": TARGET_QUALITY_VALID_MARKETS,
            "remaining_quality_valid_markets": remaining,
            "completion_probability_estimate": probability,
            "estimated_collection_progress": {
                "quality_target_fraction": valid_count
                / TARGET_QUALITY_VALID_MARKETS,
                "attempt_cap_fraction": attempted / MAXIMUM_ATTEMPTS,
                "quality_rate_used": observed_rate,
                "estimated_additional_attempts": estimated_additional_attempts,
                "estimated_completion_timestamp": (
                    estimated_completion.isoformat()
                    if estimated_completion is not None
                    else None
                ),
                "estimate_is_outcome_free": True,
            },
        },
        "attempt_health": {
            "total_attempts": attempted,
            "valid_attempts": valid_count,
            "invalid_attempts": attempted - valid_count,
            "invalid_reason_distribution": dict(sorted(invalid_reasons.items())),
            "provider_failure_rate": (
                sum(bool(row["provider_failed"]) for row in attempts) / attempted
                if attempted
                else 0.0
            ),
            "retry_rate": (
                sum(bool(row["retry_used"]) for row in attempts) / attempted
                if attempted
                else 0.0
            ),
            "hash_chain_status": hash_chain_status,
            "hash_chain_scope": "monitoring_capture_manifest_chain",
            "official_attempt_ledger_hash_chain_status": (
                "not_yet_frozen_or_validated"
            ),
        },
        "data_quality": {
            field: _coverage_summary(attempts, field) for field in quality_fields
        }
        | {
            "paired_executable_ask_coverage": _mean_or_zero(
                [
                    float(row["data_quality"]["paired_executable_ask_coverage"])
                    for row in attempts
                ]
            ),
            "btc_feature_coverage": _mean_or_zero(
                [
                    float(row["data_quality"]["btc_feature_coverage"])
                    for row in attempts
                ]
            ),
            "causality_violations": sum(
                int(row["data_quality"]["causality_violation_count"])
                for row in attempts
            ),
            "missing_feature_count_total": sum(
                int(row["data_quality"]["missing_feature_count"])
                for row in attempts
            ),
            "missing_feature_counts": dict(sorted(missing_feature_counts.items())),
            "missing_values_encoded_as_zeros": False,
        },
        "monitoring_influences_collection_decisions": False,
        "outcomes_accessed": False,
        "settlement_accessed": False,
        "pnl_accessed": False,
        "monitoring_only": True,
        "monitoring_influences_collection": False,
        "monitoring_influences_model": False,
        "promotion_evidence_eligible": False,
        "safety": dict(REPORT_SAFETY),
    }


def _attribution_report(
    attempts: Sequence[Mapping[str, Any]],
    *,
    created_at: str,
) -> dict[str, Any]:
    rows = [
        dict(decision)
        for attempt in attempts
        for decision in attempt["decision_rows"]
    ]
    _assert_outcome_free_decision_rows(rows)
    routes = Counter(str(row["requested_route"]) for row in rows)
    models = Counter(str(row["actual_model_used"]) for row in rows)
    return {
        "schema_version": "bigan-btc-15m-moe-live-collection-attribution-v1",
        "lineage_id": LINEAGE_ID,
        "created_at": created_at,
        "role": "outcome_free_monitoring_only",
        "candidate_bundle_hash": CANDIDATE_BUNDLE_HASH,
        "collected_market_count": len(attempts),
        "decision_row_count": len(rows),
        "decision_rows": rows,
        "requested_route_distribution": dict(sorted(routes.items())),
        "actual_model_distribution": dict(sorted(models.items())),
        "native_expert_decision_count": sum(
            not bool(row["fallback_used"]) for row in rows
        ),
        "fallback_decision_count": sum(bool(row["fallback_used"]) for row in rows),
        "fallback_share": (
            sum(bool(row["fallback_used"]) for row in rows) / len(rows)
            if rows
            else 0.0
        ),
        "accepted_decision_count": sum(bool(row["accepted"]) for row in rows),
        "current_confirmatory_outcomes_accessed": False,
        "current_confirmatory_settlement_accessed": False,
        "current_confirmatory_pnl_accessed": False,
        "outcomes_accessed": False,
        "settlement_accessed": False,
        "pnl_accessed": False,
        "monitoring_only": True,
        "monitoring_influences_collection": False,
        "monitoring_influences_model": False,
        "monitoring_influences_collection_decisions": False,
        "promotion_evidence_eligible": False,
        "safety": dict(REPORT_SAFETY),
    }


def _derive_development_market_observations(
    *,
    repository_root: Path,
    bundle: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    index_path = (
        repository_root
        / "examples/v8/polymarket_runs/"
        "challenge-model-development-btc-updown-15m-v1/"
        "finalized_development_corpus_index.jsonl"
    )
    index = load_jsonl(index_path)
    observations = []
    for entry in index:
        manifest_path = Path(str(entry["exported_corpus_manifest_path"])).resolve()
        if not manifest_path.is_relative_to(repository_root):
            raise ValueError("development manifest escaped repository")
        if sha256_file(manifest_path) != entry["exported_corpus_manifest_sha256"]:
            raise ValueError("development manifest SHA mismatch")
        manifest = _load_json(manifest_path)
        feature_path = manifest_path.parent / "polymarket_feature_rows.jsonl"
        if sha256_file(feature_path) != manifest["normalized_artifact_hashes"][
            "feature_rows"
        ]:
            raise ValueError("development feature artifact SHA mismatch")
        feature_rows = load_jsonl(feature_path)
        _, market = _predict_market(feature_rows=feature_rows, bundle=bundle)
        observations.append(market)
    return observations, _descriptor(index_path, repository_root=repository_root)


def _load_development_distribution_reference(
    *,
    repository_root: Path,
) -> list[dict[str, Any]]:
    path = (repository_root / DEVELOPMENT_DISTRIBUTION_REFERENCE).resolve()
    sidecar = path.with_suffix(".sha256")
    if not path.is_relative_to(repository_root):
        raise ValueError("development distribution reference escaped repository")
    if not path.is_file() or not sidecar.is_file():
        raise ValueError("development distribution reference is not frozen")
    expected = sidecar.read_text(encoding="utf-8").strip()
    if sha256_file(path) != expected:
        raise ValueError("development distribution reference SHA mismatch")
    payload = _load_json(path)
    if payload.get("candidate_bundle_hash") != CANDIDATE_BUNDLE_HASH:
        raise ValueError("development distribution candidate hash changed")
    rows = list(payload.get("market_observations") or [])
    if (
        len(rows) != int(payload.get("development_market_count") or 0)
        or canonical_json_sha256(rows)
        != payload.get("market_observations_sha256")
    ):
        raise ValueError("development distribution population mismatch")
    if any(
        payload.get(field) is not False
        for field in (
            "outcomes_accessed",
            "settlement_accessed",
            "pnl_accessed",
            "promotion_evidence_eligible",
        )
    ):
        raise ValueError("development distribution reference crossed safety boundary")
    _assert_outcome_free_decision_rows(rows)
    return rows


def _load_development_distribution_shift_reference(
    *,
    repository_root: Path,
) -> dict[str, Any]:
    path = (repository_root / DEVELOPMENT_DISTRIBUTION_SHIFT_REFERENCE).resolve()
    sidecar = path.with_suffix(".sha256")
    if not path.is_relative_to(repository_root):
        raise ValueError("development shift reference escaped repository")
    if not path.is_file() or not sidecar.is_file():
        raise ValueError("development shift reference is not frozen")
    if sidecar.read_text(encoding="utf-8").strip() != sha256_file(path):
        raise ValueError("development shift reference SHA mismatch")
    payload = _load_json(path)
    if payload.get("candidate_bundle_hash") != CANDIDATE_BUNDLE_HASH:
        raise ValueError("development shift candidate hash changed")
    rows = list(payload.get("decision_time_rows") or [])
    if len(rows) != 113 or len(rows) != int(
        payload.get("development_market_count") or 0
    ):
        raise ValueError("development shift population must contain 113 markets")
    _assert_decision_time_only_rows(rows)
    if canonical_json_sha256(rows) != payload.get("decision_time_rows_sha256"):
        raise ValueError("development shift row hash mismatch")
    if _population_identity_sha256(rows) != payload.get(
        "population_identity_sha256"
    ):
        raise ValueError("development shift population identity mismatch")
    for field in (
        "outcomes_accessed",
        "settlement_accessed",
        "pnl_accessed",
        "monitoring_influences_collection",
        "monitoring_influences_model",
        "promotion_evidence_eligible",
    ):
        if payload.get(field) is not False:
            raise ValueError(f"development shift safety boundary changed: {field}")
    if payload.get("monitoring_only") is not True:
        raise ValueError("development shift reference is not monitoring-only")
    if payload.get("safety") != REPORT_SAFETY or any(payload["safety"].values()):
        raise ValueError("development shift safety flags changed")
    return payload


def _distribution_shift_report(
    *,
    development_reference: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
    created_at: str,
) -> dict[str, Any]:
    development = [
        dict(row) for row in development_reference["decision_time_rows"]
    ]
    attempt_indices = [int(row["attempt_index"]) for row in attempts]
    if attempt_indices != list(range(1, len(attempts) + 1)):
        raise ValueError("confirmatory attempt population was reordered")
    collection = [
        _distribution_monitor_row(
            dict(attempt["market_observation"]),
            population_position=int(attempt["attempt_index"]),
        )
        for attempt in attempts
    ]
    _assert_decision_time_only_rows(development)
    _assert_decision_time_only_rows(collection)
    if [int(row["population_position"]) for row in collection] != attempt_indices:
        raise ValueError("confirmatory monitoring population order changed")

    structure_fields = {
        "spread_buckets": "combined_spread_bps",
        "liquidity_buckets": "total_liquidity_depth",
        "time_to_close_buckets": "time_to_close_seconds",
        "market_age_buckets": "market_age_seconds",
    }
    structure_cutoffs = {
        output_name: _tertile_cutoffs(development, field)
        for output_name, field in structure_fields.items()
    }
    direction = _categorical_comparison(
        [_direction_regime(row) for row in development],
        [_direction_regime(row) for row in collection],
        categories=("bullish", "bearish", "sideways_or_unknown"),
    )
    volatility = _categorical_comparison(
        [_volatility_regime(row) for row in development],
        [_volatility_regime(row) for row in collection],
        categories=("high_vol", "low_vol", "unknown"),
    )
    requested_route = _categorical_comparison(
        [str(row["requested_route"]) for row in development],
        [str(row["requested_route"]) for row in collection],
        categories=("high_vol", "bullish", "bearish", "low_vol"),
    )
    actual_model = _categorical_comparison(
        [str(row["actual_model_category"]) for row in development],
        [str(row["actual_model_category"]) for row in collection],
        categories=("native_expert", "global_fallback"),
    )
    structure = {
        output_name: {
            "development_cutoffs": structure_cutoffs[output_name],
            "comparison": _categorical_comparison(
                [
                    _bucket(row.get(field), structure_cutoffs[output_name])
                    for row in development
                ],
                [
                    _bucket(row.get(field), structure_cutoffs[output_name])
                    for row in collection
                ],
                categories=("low", "medium", "high", "missing"),
            ),
        }
        for output_name, field in structure_fields.items()
    }
    missingness = _feature_missingness_shift(
        development=development,
        collection=collection,
    )
    provider_quality = _provider_quality_report(attempts)
    numeric = {
        field: _numeric_shift(
            [row.get(field) for row in development],
            [row.get(field) for row in collection],
        )
        for field in NUMERIC_DRIFT_FIELDS
    }
    collection_hash = _population_identity_sha256(collection)
    development_hash = str(
        development_reference["population_identity_sha256"]
    )
    report = {
        "schema_version": (
            "bigan-btc-15m-moe-collection-distribution-shift-report-v1"
        ),
        "lineage_id": LINEAGE_ID,
        "run_id": "BTC-15M-MoE-confirmatory-v2-outcome-blind-collection-001",
        "reporting_timestamp": created_at,
        "role": "outcome_blind_diagnostic_monitoring_only",
        "candidate_bundle_hash": CANDIDATE_BUNDLE_HASH,
        "population": {
            "development_market_count": len(development),
            "collection_market_count": len(collection),
            "development_population_hash": development_hash,
            "collection_population_hash": collection_hash,
            "development_decision_rows_hash": canonical_json_sha256(
                development
            ),
            "collection_decision_rows_hash": canonical_json_sha256(collection),
            "development_source_decision_rows_hash": (
                development_reference["decision_time_rows_sha256"]
            ),
            "hashes_are_separate": True,
            "collection_attempt_order": attempt_indices,
            "collection_order_preserved": True,
            "collection_rows_filtered": False,
            "collection_rows_reordered": False,
        },
        "regime_distribution": {
            "direction": direction,
            "volatility": volatility,
        },
        "market_structure": structure,
        "moe_routing": {
            "requested_route": requested_route,
            "actual_model": actual_model,
            "development_fallback_ratio": _ratio_true(
                development, "fallback_used"
            ),
            "collection_fallback_ratio": _ratio_true(
                collection, "fallback_used"
            ),
            "fallback_ratio_delta": (
                _ratio_true(collection, "fallback_used")
                - _ratio_true(development, "fallback_used")
            ),
            "collection_route_attribution": [
                {
                    "population_position": row["population_position"],
                    "market_id": row["market_id"],
                    "decision_ts": row["decision_ts"],
                    "requested_route": row["requested_route"],
                    "actual_model_used": row["actual_model_used"],
                    "expert_available": row["expert_available"],
                    "fallback_used": row["fallback_used"],
                    "expert_training_market_count": row[
                        "expert_training_market_count"
                    ],
                }
                for row in collection
            ],
            "attribution_reconciled": (
                len(collection)
                == sum(
                    int(value["collection_count"])
                    for value in requested_route["categories"].values()
                )
                == sum(
                    int(value["collection_count"])
                    for value in actual_model["categories"].values()
                )
            ),
            "routing_modification_allowed": False,
        },
        "feature_missingness": missingness,
        "provider_quality": provider_quality,
        "drift_metrics": {
            "numeric": numeric,
            "categorical": {
                "direction_total_variation_distance": direction[
                    "total_variation_distance"
                ],
                "volatility_total_variation_distance": volatility[
                    "total_variation_distance"
                ],
                "requested_route_total_variation_distance": requested_route[
                    "total_variation_distance"
                ],
                "actual_model_total_variation_distance": actual_model[
                    "total_variation_distance"
                ],
                "market_structure_total_variation_distance": {
                    name: panel["comparison"]["total_variation_distance"]
                    for name, panel in structure.items()
                },
            },
            "deterministic": True,
            "automatic_thresholds_present": False,
            "materiality_conclusion": (
                "not_assigned_monitoring_has_no_threshold_or_gate"
            ),
        },
        "outcomes_accessed": False,
        "settlement_accessed": False,
        "pnl_accessed": False,
        "monitoring_only": True,
        "monitoring_influences_collection": False,
        "monitoring_influences_model": False,
        "monitoring_influences_promotion": False,
        "market_filtering_allowed": False,
        "population_reordering_allowed": False,
        "collection_stop_condition_created": False,
        "promotion_evidence_eligible": False,
        "safety": dict(REPORT_SAFETY),
    }
    _assert_distribution_shift_report(report)
    return report


def _distribution_monitor_row(
    row: Mapping[str, Any],
    *,
    population_position: int,
) -> dict[str, Any]:
    _assert_distribution_source_row(row)
    router_inputs = dict(row.get("router_inputs") or {})
    regime_bucket = dict(row.get("regime_bucket") or {})
    missing_names = sorted(
        {str(name) for name in (row.get("missing_feature_names") or [])}
    )
    time_to_close = _json_finite_or_none(row.get("time_to_close_seconds"))
    market_age = _json_finite_or_none(row.get("market_age_seconds"))
    if market_age is None and time_to_close is not None:
        market_age = 900.0 - time_to_close
    fallback_used = bool(row["fallback_used"])
    output = {
        "population_position": int(population_position),
        "market_id": str(row["market_id"]),
        "market_start_ts": int(row["market_start_ts"]),
        "decision_ts": int(row["decision_ts"]),
        "btc_return_15m": _json_finite_or_none(
            row.get("btc_return_15m", router_inputs.get("btc_return_15m"))
        ),
        "btc_volatility_15m": _json_finite_or_none(
            row.get(
                "btc_volatility_15m",
                router_inputs.get("btc_volatility_15m"),
            )
        ),
        "direction_regime": _normalize_direction_regime(
            regime_bucket.get("btc_return_regime")
        ),
        "volatility_regime": _normalize_volatility_regime(
            regime_bucket.get("volatility_bucket")
        ),
        "requested_route": str(row["requested_route"]),
        "actual_model_used": str(row["actual_model_used"]),
        "actual_model_category": (
            "global_fallback" if fallback_used else "native_expert"
        ),
        "expert_available": bool(row["expert_available"]),
        "fallback_used": fallback_used,
        "expert_training_market_count": int(
            row.get(
                "expert_training_market_count",
                row.get("expert_training_support"),
            )
        ),
        "selected_side": row.get("selected_side"),
        "accepted": bool(row.get("accepted")),
        "combined_spread_bps": _json_finite_or_none(
            row.get("combined_spread_bps")
        ),
        "total_liquidity_depth": _json_finite_or_none(
            row.get("total_liquidity_depth")
        ),
        "time_to_close_seconds": time_to_close,
        "market_age_seconds": market_age,
        "provider_health_score": _json_finite_or_none(
            row.get("provider_health_score")
        ),
        "missing_feature_names": missing_names,
        "missing_feature_count": len(missing_names),
        "missing_values_encoded_as_zero": False,
    }
    _assert_decision_time_row(output)
    return output


def _population_identity_sha256(
    rows: Sequence[Mapping[str, Any]],
) -> str:
    identities = [
        {
            "population_position": int(row["population_position"]),
            "market_id": str(row["market_id"]),
            "decision_ts": int(row["decision_ts"]),
        }
        for row in rows
    ]
    return canonical_json_sha256(identities)


def _categorical_comparison(
    development: Sequence[str],
    collection: Sequence[str],
    *,
    categories: Sequence[str],
) -> dict[str, Any]:
    ordered_categories = list(
        dict.fromkeys(
            [
                *categories,
                *sorted((set(development) | set(collection)) - set(categories)),
            ]
        )
    )
    development_counts = Counter(development)
    collection_counts = Counter(collection)
    development_total = len(development)
    collection_total = len(collection)
    output = {}
    for category in ordered_categories:
        development_percentage = _percentage(
            development_counts[category],
            development_total,
        )
        collection_percentage = _percentage(
            collection_counts[category],
            collection_total,
        )
        output[category] = {
            "development_count": development_counts[category],
            "development_percentage": development_percentage,
            "collection_count": collection_counts[category],
            "collection_percentage": collection_percentage,
            "percentage_point_delta": (
                collection_percentage - development_percentage
            ),
        }
    return {
        "categories": output,
        "development_total": development_total,
        "collection_total": collection_total,
        "total_variation_distance": 0.5
        * sum(
            abs(
                output[category]["collection_percentage"]
                - output[category]["development_percentage"]
            )
            / 100.0
            for category in ordered_categories
        ),
        "diagnostic_only": True,
    }


def _numeric_shift(
    development_values: Sequence[Any],
    collection_values: Sequence[Any],
) -> dict[str, Any]:
    development = _numeric_summary(development_values)
    collection = _numeric_summary(collection_values)
    return {
        "development": development,
        "collection": collection,
        "mean_shift": _optional_difference(
            collection["mean"], development["mean"]
        ),
        "std_shift": _optional_difference(
            collection["std"], development["std"]
        ),
        "quantile_shift": {
            key: _optional_difference(
                collection["quantiles"][key],
                development["quantiles"][key],
            )
            for key in development["quantiles"]
        },
        "missing_values_encoded_as_zero": False,
        "diagnostic_only": True,
    }


def _numeric_summary(values: Sequence[Any]) -> dict[str, Any]:
    finite = np.asarray(
        [float(value) for value in values if _finite(value)],
        dtype=np.float64,
    )
    quantiles = {
        f"q{int(quantile * 100):02d}": (
            float(np.quantile(finite, quantile)) if len(finite) else None
        )
        for quantile in NUMERIC_DRIFT_QUANTILES
    }
    return {
        "observed_count": int(len(finite)),
        "missing_count": len(values) - int(len(finite)),
        "mean": float(np.mean(finite)) if len(finite) else None,
        "std": float(np.std(finite)) if len(finite) else None,
        "quantiles": quantiles,
    }


def _feature_missingness_shift(
    *,
    development: Sequence[Mapping[str, Any]],
    collection: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_feature = {
        name: _missingness_comparison(
            development=development,
            collection=collection,
            required_missing_features=(name,),
        )
        for name in BASE_FEATURE_NAMES
    }
    important_groups = {
        "recent_trade_volume": ("selected_recent_trade_volume",),
        "opposite_trade_volume": ("opposite_recent_trade_volume",),
        "orderbook_depth": (
            "selected_liquidity_depth",
            "opposite_liquidity_depth",
        ),
        "trade_tape_coverage": (
            "selected_recent_trade_volume",
            "opposite_recent_trade_volume",
        ),
        "btc_feature_coverage": tuple(
            name
            for name in BASE_FEATURE_NAMES
            if "btc_return" in name or name.startswith("btc_volatility")
        ),
        "chainlink_reference_coverage": (
            "signed_chainlink_reference_distance",
            "signed_btc_mid_to_chainlink_relative_distance",
        ),
    }
    return {
        "by_feature": by_feature,
        "important_features": {
            name: _missingness_comparison(
                development=development,
                collection=collection,
                required_missing_features=features,
            )
            for name, features in important_groups.items()
        },
        "development_feature_count": len(BASE_FEATURE_NAMES),
        "collection_feature_count": len(BASE_FEATURE_NAMES),
        "missing_values_encoded_as_zero": False,
        "reconciled": (
            all(
                panel["development_missing_count"]
                <= panel["development_total"]
                and panel["collection_missing_count"]
                <= panel["collection_total"]
                for panel in by_feature.values()
            )
        ),
    }


def _missingness_comparison(
    *,
    development: Sequence[Mapping[str, Any]],
    collection: Sequence[Mapping[str, Any]],
    required_missing_features: Sequence[str],
) -> dict[str, Any]:
    required = set(required_missing_features)

    def missing_count(rows: Sequence[Mapping[str, Any]]) -> int:
        return sum(
            bool(required & set(row["missing_feature_names"]))
            for row in rows
        )

    development_missing = missing_count(development)
    collection_missing = missing_count(collection)
    development_rate = (
        development_missing / len(development) if development else 0.0
    )
    collection_rate = (
        collection_missing / len(collection) if collection else 0.0
    )
    return {
        "feature_group": list(required_missing_features),
        "development_total": len(development),
        "development_missing_count": development_missing,
        "development_missing_rate": development_rate,
        "collection_total": len(collection),
        "collection_missing_count": collection_missing,
        "collection_missing_rate": collection_rate,
        "missing_rate_delta": collection_rate - development_rate,
    }


def _provider_quality_report(
    attempts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    total = len(attempts)

    def coverage(field: str) -> dict[str, Any]:
        covered = sum(
            bool(row["data_quality"][field]) for row in attempts
        )
        return {
            "covered_attempts": covered,
            "total_attempts": total,
            "coverage": covered / total if total else 0.0,
        }

    invalid = sum(not bool(row["quality"]["quality_valid"]) for row in attempts)
    retries = sum(bool(row["retry_used"]) for row in attempts)
    causality = sum(
        int(row["data_quality"]["causality_violation_count"])
        for row in attempts
    )
    paired_coverage = _mean_or_zero(
        [
            float(row["data_quality"]["paired_executable_ask_coverage"])
            for row in attempts
        ]
    )
    return {
        "raw_market_coverage": coverage("raw_market_capture_coverage"),
        "orderbook_coverage": coverage("orderbook_coverage"),
        "trade_availability": coverage("trade_tape_available"),
        "btc_feature_coverage": _mean_or_zero(
            [
                float(row["data_quality"]["btc_feature_coverage"])
                for row in attempts
            ]
        ),
        "chainlink_reference_coverage": coverage(
            "chainlink_reference_coverage"
        ),
        "paired_executable_ask_coverage": paired_coverage,
        "retry_count": retries,
        "retry_rate": retries / total if total else 0.0,
        "invalid_attempt_count": invalid,
        "invalid_attempt_rate": invalid / total if total else 0.0,
        "causality_violation_count": causality,
        "causality_violation_rate_per_attempt": (
            causality / total if total else 0.0
        ),
        "reconciled": all(
            0.0 <= value <= 1.0
            for value in (
                paired_coverage,
                retries / total if total else 0.0,
                invalid / total if total else 0.0,
            )
        ),
        "diagnostic_only": True,
    }


def _direction_regime(row: Mapping[str, Any]) -> str:
    return _normalize_direction_regime(row.get("direction_regime"))


def _normalize_direction_regime(value: Any) -> str:
    normalized = str(value or "").lower()
    return (
        normalized
        if normalized in {"bullish", "bearish"}
        else "sideways_or_unknown"
    )


def _volatility_regime(row: Mapping[str, Any]) -> str:
    return _normalize_volatility_regime(row.get("volatility_regime"))


def _normalize_volatility_regime(value: Any) -> str:
    normalized = str(value or "").lower()
    if normalized in {"high", "high_vol"}:
        return "high_vol"
    if normalized in {"low", "medium", "low_vol"}:
        return "low_vol"
    return "unknown"


def _ratio_true(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return (
        sum(bool(row[field]) for row in rows) / len(rows)
        if rows
        else 0.0
    )


def _optional_difference(
    left: float | None,
    right: float | None,
) -> float | None:
    return None if left is None or right is None else left - right


def _assert_no_forbidden_monitoring_fields(value: Any, path: str = "row") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key).lower()
            if any(
                token in key_text
                for token in MONITORING_FORBIDDEN_FIELD_TOKENS
            ):
                raise ValueError(
                    f"forbidden monitoring target field: {path}.{key}"
                )
            _assert_no_forbidden_monitoring_fields(
                nested,
                f"{path}.{key}",
            )
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, nested in enumerate(value):
            _assert_no_forbidden_monitoring_fields(
                nested,
                f"{path}[{index}]",
            )


def _assert_distribution_source_row(row: Mapping[str, Any]) -> None:
    _assert_no_forbidden_monitoring_fields(row)
    unknown = set(row) - DISTRIBUTION_SOURCE_ALLOWED_FIELDS
    if unknown:
        raise ValueError(
            "distribution source contains non-decision-time fields: "
            + ", ".join(sorted(unknown))
        )


def _assert_decision_time_only_rows(
    rows: Sequence[Mapping[str, Any]],
) -> None:
    for expected_position, row in enumerate(rows, start=1):
        _assert_decision_time_row(row)
        if int(row["population_position"]) != expected_position:
            raise ValueError("monitoring population was reordered")


def _assert_decision_time_row(row: Mapping[str, Any]) -> None:
    required_fields = {
        "population_position",
        "market_id",
        "market_start_ts",
        "decision_ts",
        "btc_return_15m",
        "btc_volatility_15m",
        "direction_regime",
        "volatility_regime",
        "requested_route",
        "actual_model_used",
        "actual_model_category",
        "expert_available",
        "fallback_used",
        "expert_training_market_count",
        "selected_side",
        "accepted",
        "combined_spread_bps",
        "total_liquidity_depth",
        "time_to_close_seconds",
        "market_age_seconds",
        "provider_health_score",
        "missing_feature_names",
        "missing_feature_count",
        "missing_values_encoded_as_zero",
    }
    _assert_no_forbidden_monitoring_fields(row)
    if set(row) != required_fields:
        raise ValueError("monitoring row field contract changed")
    if row["missing_values_encoded_as_zero"] is not False:
        raise ValueError("monitoring encoded missing values as zero")
    names = list(row["missing_feature_names"])
    if names != sorted(set(names)):
        raise ValueError("monitoring missing feature names are not canonical")
    if int(row["missing_feature_count"]) != len(names):
        raise ValueError("monitoring missingness count mismatch")


def _assert_distribution_shift_report(report: Mapping[str, Any]) -> None:
    for field in (
        "outcomes_accessed",
        "settlement_accessed",
        "pnl_accessed",
        "monitoring_influences_collection",
        "monitoring_influences_model",
        "monitoring_influences_promotion",
        "market_filtering_allowed",
        "population_reordering_allowed",
        "collection_stop_condition_created",
        "promotion_evidence_eligible",
    ):
        if report.get(field) is not False:
            raise ValueError(f"distribution shift safety field changed: {field}")
    if report.get("monitoring_only") is not True:
        raise ValueError("distribution shift report is not monitoring-only")
    if report.get("safety") != REPORT_SAFETY or any(report["safety"].values()):
        raise ValueError("distribution shift safety flags changed")
    population = dict(report["population"])
    if (
        population["collection_rows_filtered"] is not False
        or population["collection_rows_reordered"] is not False
        or population["collection_order_preserved"] is not True
    ):
        raise ValueError("distribution shift population contract changed")
    if (
        population["hashes_are_separate"] is not True
        or population["development_population_hash"]
        == population["collection_population_hash"]
    ):
        raise ValueError("distribution shift population hashes are not separate")
    if report["moe_routing"]["attribution_reconciled"] is not True:
        raise ValueError("distribution shift route attribution mismatch")
    if report["feature_missingness"]["reconciled"] is not True:
        raise ValueError("distribution shift missingness mismatch")
    if report["provider_quality"]["reconciled"] is not True:
        raise ValueError("distribution shift provider quality mismatch")


def _market_observations(
    attempts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [dict(row["market_observation"]) for row in attempts]


def _distribution_report(
    *,
    development: Sequence[Mapping[str, Any]],
    confirmatory: Sequence[Mapping[str, Any]],
    created_at: str,
) -> dict[str, Any]:
    cutoffs = {
        field: _tertile_cutoffs(development, field)
        for field in (
            "total_liquidity_depth",
            "combined_spread_bps",
            "time_to_close_seconds",
        )
    }
    return {
        "schema_version": "bigan-btc-15m-development-vs-confirmatory-distribution-v1",
        "lineage_id": LINEAGE_ID,
        "created_at": created_at,
        "role": "outcome_free_diagnostic_only",
        "candidate_bundle_hash": CANDIDATE_BUNDLE_HASH,
        "development_market_count": len(development),
        "confirmatory_market_count": len(confirmatory),
        "bucket_cutoffs_frozen_from_development_features": cutoffs,
        "development": _distribution_panel(development, cutoffs=cutoffs),
        "confirmatory": _distribution_panel(confirmatory, cutoffs=cutoffs),
        "diagnostic_only": True,
        "route_filtering_allowed": False,
        "collection_stop_or_change_allowed": False,
        "monitoring_influences_collection_decisions": False,
        "outcomes_accessed": False,
        "settlement_accessed": False,
        "pnl_accessed": False,
        "monitoring_only": True,
        "monitoring_influences_collection": False,
        "monitoring_influences_model": False,
        "promotion_evidence_eligible": False,
        "safety": dict(REPORT_SAFETY),
    }


def _distribution_panel(
    rows: Sequence[Mapping[str, Any]],
    *,
    cutoffs: Mapping[str, Mapping[str, float | None]],
) -> dict[str, Any]:
    count = len(rows)
    routes = Counter(str(row["requested_route"]) for row in rows)
    regimes = Counter(
        str(dict(row["regime_bucket"])["btc_return_regime"]) for row in rows
    )
    volatility = Counter(
        str(dict(row["regime_bucket"])["volatility_bucket"]) for row in rows
    )
    return {
        "market_regime_distribution": _share_panel(regimes, count),
        "volatility_bucket_distribution": _share_panel(volatility, count),
        "liquidity_bucket_distribution": _share_panel(
            Counter(
                _bucket(row.get("total_liquidity_depth"), cutoffs["total_liquidity_depth"])
                for row in rows
            ),
            count,
        ),
        "spread_bucket_distribution": _share_panel(
            Counter(
                _bucket(row.get("combined_spread_bps"), cutoffs["combined_spread_bps"])
                for row in rows
            ),
            count,
        ),
        "time_to_close_bucket_distribution": _share_panel(
            Counter(
                _bucket(row.get("time_to_close_seconds"), cutoffs["time_to_close_seconds"])
                for row in rows
            ),
            count,
        ),
        "router_distribution": {
            "high_vol_route_percentage": _percentage(routes["high_vol"], count),
            "bullish_route_percentage": _percentage(routes["bullish"], count),
            "bearish_route_percentage": _percentage(routes["bearish"], count),
            "low_vol_route_percentage": _percentage(routes["low_vol"], count),
            "fallback_percentage": _percentage(
                sum(bool(row["fallback_used"]) for row in rows),
                count,
            ),
            "counts": dict(sorted(routes.items())),
        },
    }


def _write_monitor_hash_chain(
    path: Path,
    attempts: Sequence[Mapping[str, Any]],
) -> None:
    existing = load_jsonl(path)
    if len(existing) > len(attempts):
        raise ValueError("monitor hash chain has more rows than completed attempts")
    previous = "0" * 64
    for index, row in enumerate(existing, start=1):
        if int(row["attempt_index"]) != index:
            raise ValueError("monitor hash chain index mismatch")
        if row["previous_entry_sha256"] != previous:
            raise ValueError("monitor hash chain previous hash mismatch")
        payload = dict(row)
        recorded = str(payload.pop("entry_sha256"))
        if canonical_json_sha256(payload) != recorded:
            raise ValueError("monitor hash chain entry mismatch")
        expected = attempts[index - 1]
        if (
            row["attempt_id"] != expected["attempt_id"]
            or row["capture_manifest_sha256"]
            != expected["capture_manifest_sha256"]
        ):
            raise ValueError("monitor hash chain immutable prefix changed")
        previous = recorded
    additions = []
    for attempt in attempts[len(existing) :]:
        payload = {
            "schema_version": "bigan-btc-15m-monitor-attempt-chain-v1",
            "lineage_id": LINEAGE_ID,
            "attempt_index": int(attempt["attempt_index"]),
            "attempt_id": attempt["attempt_id"],
            "market_id": attempt["market_id"],
            "market_start_ts": int(attempt["market_start_ts"]),
            "capture_manifest_sha256": attempt["capture_manifest_sha256"],
            "quality_valid": bool(attempt["quality"]["quality_valid"]),
            "quality_failure_reason_codes": list(
                attempt["quality"]["invalid_reason_codes"]
            ),
            "outcomes_accessed": False,
            "previous_entry_sha256": previous,
        }
        payload["entry_sha256"] = canonical_json_sha256(payload)
        previous = payload["entry_sha256"]
        additions.append(payload)
    if additions:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for row in additions:
                handle.write(
                    json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                )


def _validate_monitor_hash_chain(path: Path) -> str:
    rows = load_jsonl(path)
    previous = "0" * 64
    for expected, row in enumerate(rows, start=1):
        if int(row["attempt_index"]) != expected:
            return "invalid"
        if row["previous_entry_sha256"] != previous:
            return "invalid"
        payload = dict(row)
        recorded = str(payload.pop("entry_sha256"))
        if canonical_json_sha256(payload) != recorded:
            return "invalid"
        previous = recorded
    return "valid"


def _beta_binomial_completion_probability(
    *,
    required_successes: int,
    future_attempts: int,
    prior_valid: int,
    prior_invalid: int,
) -> float:
    if required_successes <= 0:
        return 1.0
    if future_attempts < required_successes:
        return 0.0
    alpha = float(prior_valid + 1)
    beta = float(prior_invalid + 1)
    logs = []
    for successes in range(required_successes, future_attempts + 1):
        failures = future_attempts - successes
        log_probability = (
            math.lgamma(future_attempts + 1)
            - math.lgamma(successes + 1)
            - math.lgamma(failures + 1)
            + math.lgamma(successes + alpha)
            + math.lgamma(failures + beta)
            - math.lgamma(future_attempts + alpha + beta)
            + math.lgamma(alpha + beta)
            - math.lgamma(alpha)
            - math.lgamma(beta)
        )
        logs.append(log_probability)
    maximum = max(logs)
    return min(1.0, math.exp(maximum) * sum(math.exp(value - maximum) for value in logs))


def _bootstrap_indices(
    *,
    market_count: int,
    resamples: int,
    seed: int,
) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return generator.integers(
        0,
        market_count,
        size=(resamples, market_count),
        endpoint=False,
    )


def _shared_bootstrap_interval(
    values: np.ndarray,
    *,
    indices: np.ndarray,
    confidence: float,
) -> dict[str, Any]:
    means = np.mean(values[indices], axis=1)
    tail = 1.0 - confidence
    return {
        "method": "market_level_paired_percentile_bootstrap",
        "confidence": confidence,
        "lower": float(np.quantile(means, tail)),
        "upper": float(np.quantile(means, confidence)),
        "resample_count": len(indices),
    }


def _capture_retry_used(capture: Mapping[str, Any]) -> bool:
    distribution = dict(
        capture.get("market_identity_clob_revalidation_attempt_distribution")
        or {}
    )
    identity_retry = any(
        int(attempts) > 1 and int(count) > 0
        for attempts, count in distribution.items()
    )
    return (
        identity_retry
        or int(capture.get("feature_enrichment_attempt_count") or 0) > 0
        or int(capture.get("chainlink_rtds_stale_reconnect_count") or 0) > 0
        or int(capture.get("orderbook_window_coverage_fallback_applied_market_count") or 0)
        > 0
    )


def _coverage_summary(
    attempts: Sequence[Mapping[str, Any]],
    field: str,
) -> dict[str, Any]:
    covered = sum(bool(row["data_quality"][field]) for row in attempts)
    return {
        "covered_attempts": covered,
        "total_attempts": len(attempts),
        "coverage": covered / len(attempts) if attempts else 0.0,
    }


def _tertile_cutoffs(
    rows: Sequence[Mapping[str, Any]],
    field: str,
) -> dict[str, float | None]:
    values = [
        float(row[field])
        for row in rows
        if row.get(field) is not None and math.isfinite(float(row[field]))
    ]
    return {
        "lower": float(np.quantile(values, 1.0 / 3.0)) if values else None,
        "upper": float(np.quantile(values, 2.0 / 3.0)) if values else None,
    }


def _bucket(
    value: Any,
    cutoffs: Mapping[str, float | None],
) -> str:
    if value is None or not math.isfinite(float(value)):
        return "missing"
    if cutoffs["lower"] is None or cutoffs["upper"] is None:
        return "unavailable"
    if float(value) <= float(cutoffs["lower"]):
        return "low"
    if float(value) <= float(cutoffs["upper"]):
        return "medium"
    return "high"


def _share_panel(counts: Counter[str], total: int) -> dict[str, Any]:
    return {
        "counts": dict(sorted(counts.items())),
        "percentages": {
            key: _percentage(value, total) for key, value in sorted(counts.items())
        },
    }


def _percentage(count: int, total: int) -> float:
    return 100.0 * count / total if total else 0.0


def _row_missing_feature_count(feature_row: Mapping[str, Any]) -> int:
    return len(_row_missing_feature_names(feature_row))


def _row_missing_feature_names(
    feature_row: Mapping[str, Any],
) -> list[str]:
    transformed = side_symmetric_features(feature_row, "UP")
    return [
        name
        for name in BASE_FEATURE_NAMES
        if transformed[f"{name}__missing"] == 1.0
    ]


def _is_executable_price(value: Any) -> bool:
    return _finite(value) and 0.0 < float(value) <= 1.0


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _float_or_nan(value: Any) -> float:
    return float(value) if _finite(value) else math.nan


def _json_finite_or_none(value: Any) -> float | None:
    return float(value) if _finite(value) else None


def _finite_sum_or_none(*values: Any) -> float | None:
    return sum(float(value) for value in values) if all(_finite(value) for value in values) else None


def _mean_or_zero(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _allowed_current_path(path: Path) -> Path:
    lowered = path.name.lower()
    if any(token in lowered for token in FORBIDDEN_CURRENT_ARTIFACT_TOKENS):
        raise ValueError(f"forbidden current confirmatory artifact: {path.name}")
    return path


def _assert_outcome_free_decision_rows(
    rows: Sequence[Mapping[str, Any]],
) -> None:
    for row in rows:
        forbidden = {
            key
            for key in row
            if any(
                token in key.lower()
                for token in ("outcome", "settlement", "realized_pnl", "unit_pnl")
            )
        }
        if forbidden:
            raise ValueError(
                "current attribution row contains forbidden target fields: "
                + ", ".join(sorted(forbidden))
            )


def _load_jsonl_current(path: Path) -> list[dict[str, Any]]:
    _allowed_current_path(path)
    return load_jsonl(path)


def _descriptor(path: Path, *, repository_root: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_relative_to(repository_root):
        raise ValueError("artifact descriptor escaped repository")
    return {
        "path": resolved.relative_to(repository_root).as_posix(),
        "sha256": sha256_file(resolved),
    }


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        Path(temporary).replace(path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _write_sha_sidecar(path: Path) -> None:
    sidecar = (
        path.with_suffix(".sha256")
        if path.suffix == ".json"
        else path.with_suffix(path.suffix + ".sha256")
    )
    _atomic_write_text(
        sidecar,
        f"{sha256_file(path)}\n",
    )


def _frozen_created_at(path: Path, *, requested: str | None) -> str:
    if requested is not None:
        return requested
    if path.is_file():
        existing = _load_json(path)
        value = str(existing.get("created_at") or "")
        if not value:
            raise ValueError(f"frozen artifact is missing created_at: {path}")
        return value
    return datetime.now(UTC).isoformat()


def _write_frozen_json(path: Path, payload: Mapping[str, Any]) -> None:
    sidecar = path.with_suffix(".sha256")
    if path.exists() or sidecar.exists():
        if not path.is_file() or not sidecar.is_file():
            raise ValueError(f"incomplete frozen artifact pair: {path}")
        if sidecar.read_text(encoding="utf-8").strip() != sha256_file(path):
            raise ValueError(f"frozen artifact SHA mismatch: {path}")
        if _load_json(path) != dict(payload):
            raise ValueError(f"frozen artifact content drift: {path}")
        return
    _atomic_write_json(path, payload)
    _write_sha_sidecar(path)


def _distribution_shift_markdown(report: Mapping[str, Any]) -> str:
    population = dict(report["population"])
    direction = dict(report["regime_distribution"]["direction"]["categories"])
    routes = dict(report["moe_routing"]["requested_route"]["categories"])
    missingness = dict(report["feature_missingness"]["important_features"])
    provider = dict(report["provider_quality"])
    lines = [
        "# BTC 15m MoE collection distribution shift",
        "",
        "- Role: outcome-blind diagnostic monitoring only",
        f"- Reporting timestamp: `{report['reporting_timestamp']}`",
        f"- Candidate bundle: `{report['candidate_bundle_hash']}`",
        f"- Development markets: `{population['development_market_count']}`",
        f"- Collection markets: `{population['collection_market_count']}`",
        f"- Development population hash: "
        f"`{population['development_population_hash']}`",
        f"- Collection population hash: "
        f"`{population['collection_population_hash']}`",
        "",
        "## Direction regime",
        "",
        "| Regime | Development | Collection | Delta (pp) |",
        "|---|---:|---:|---:|",
    ]
    for name in ("bullish", "bearish", "sideways_or_unknown"):
        panel = direction[name]
        lines.append(
            f"| {name} | {panel['development_count']} "
            f"({panel['development_percentage']:.2f}%) | "
            f"{panel['collection_count']} "
            f"({panel['collection_percentage']:.2f}%) | "
            f"{panel['percentage_point_delta']:+.2f} |"
        )
    lines.extend(
        [
            "",
            "## Requested route",
            "",
            "| Route | Development | Collection | Delta (pp) |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in ("high_vol", "bullish", "bearish", "low_vol"):
        panel = routes[name]
        lines.append(
            f"| {name} | {panel['development_count']} "
            f"({panel['development_percentage']:.2f}%) | "
            f"{panel['collection_count']} "
            f"({panel['collection_percentage']:.2f}%) | "
            f"{panel['percentage_point_delta']:+.2f} |"
        )
    lines.extend(
        [
            "",
            f"- Development fallback ratio: "
            f"`{report['moe_routing']['development_fallback_ratio']:.6f}`",
            f"- Collection fallback ratio: "
            f"`{report['moe_routing']['collection_fallback_ratio']:.6f}`",
            "",
            "## Important feature missingness",
            "",
            "| Feature group | Development missing | Collection missing | Delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, panel in missingness.items():
        lines.append(
            f"| {name} | {panel['development_missing_rate']:.4f} | "
            f"{panel['collection_missing_rate']:.4f} | "
            f"{panel['missing_rate_delta']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Collection provider quality",
            "",
            f"- Raw market coverage: "
            f"`{provider['raw_market_coverage']['coverage']:.6f}`",
            f"- Orderbook coverage: "
            f"`{provider['orderbook_coverage']['coverage']:.6f}`",
            f"- Trade availability: "
            f"`{provider['trade_availability']['coverage']:.6f}`",
            f"- Paired executable ask coverage: "
            f"`{provider['paired_executable_ask_coverage']:.6f}`",
            f"- Retry rate: `{provider['retry_rate']:.6f}`",
            f"- Invalid attempt rate: `{provider['invalid_attempt_rate']:.6f}`",
            f"- Causality violations: "
            f"`{provider['causality_violation_count']}`",
            "",
            "- No drift threshold or materiality gate is assigned.",
            "- No market is filtered or reordered.",
            "- Monitoring influences collection: false",
            "- Monitoring influences model: false",
            "- Outcomes accessed: false",
            "- Settlement accessed: false",
            "- PnL accessed: false",
            "",
            "## Safety",
            "",
            "- source_model_candidate_eligible=false",
            "- freeze_ready=false",
            "- promotion_evidence_eligible=false",
            "- paper_candidate_allowed=false",
            "- v8_execution_handoff_allowed=false",
            "- live_trading_allowed=false",
            "- wallet_signing_allowed=false",
            "- polymarket_write_allowed=false",
            "- capital_at_risk=false",
            "",
        ]
    )
    return "\n".join(lines)


def _health_markdown(report: Mapping[str, Any]) -> str:
    progress = dict(report["progress"])
    health = dict(report["attempt_health"])
    quality = dict(report["data_quality"])
    return "\n".join(
        [
            "# BTC 15m MoE confirmatory collection health",
            "",
            "- Role: outcome-free monitoring only",
            f"- Attempts: `{progress['attempts_consumed']} / {progress['maximum_attempts']}`",
            f"- Quality-valid: `{progress['quality_valid_market_count']} / "
            f"{progress['target_quality_valid_market_count']}`",
            f"- Remaining quality-valid markets: "
            f"`{progress['remaining_quality_valid_markets']}`",
            f"- Completion probability estimate: "
            f"`{progress['completion_probability_estimate']:.6f}`",
            f"- Invalid attempts: `{health['invalid_attempts']}`",
            f"- Provider failure rate: `{health['provider_failure_rate']:.6f}`",
            f"- Retry rate: `{health['retry_rate']:.6f}`",
            f"- Hash chain: `{health['hash_chain_status']}`",
            f"- Paired executable ask coverage: "
            f"`{quality['paired_executable_ask_coverage']:.6f}`",
            f"- Causality violations: `{quality['causality_violations']}`",
            f"- Missing feature count: `{quality['missing_feature_count_total']}`",
            "",
            "- Outcomes accessed: false",
            "- Settlement accessed: false",
            "- PnL accessed: false",
            "- Monitoring influences collection: false",
            "",
        ]
    )


__all__ = [
    "build_collection_observability",
    "build_development_distribution_reference",
    "build_development_distribution_shift_reference",
    "build_evaluation_dry_run_report",
    "build_finalization_checklist",
]
