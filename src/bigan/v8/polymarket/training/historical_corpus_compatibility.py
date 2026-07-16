"""Outcome-blind compatibility audit for historical Polymarket corpora."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.contracts import compact_safety_fields
from bigan.v8.polymarket.training.execution_layer_v2_pairwise_action_advantage_lcb import (
    validate_pairwise_action_advantage_lcb_feature_contract,
    validate_pairwise_action_advantage_lcb_protocol,
)

SCHEMA_PREFIX = "bigan-v8-historical-corpus-compatibility"
CURRENT_CORPUS_SCHEMA_VERSION = "bigan-v8-polymarket-corpus-v3"
CURRENT_SBC_LABEL_SCHEMA_VERSION = (
    "bigan-v8-polymarket-sell-before-close-executable-exit-v1"
)
HISTORICAL_DEVELOPMENT_COMPATIBLE = "historical_development_compatible"
HISTORICAL_DEVELOPMENT_CONVERTIBLE = "historical_development_convertible"
HISTORICAL_INCOMPATIBLE = "historical_incompatible"
CLASSIFICATION_PRIORITY = {
    HISTORICAL_INCOMPATIBLE: 0,
    HISTORICAL_DEVELOPMENT_CONVERTIBLE: 1,
    HISTORICAL_DEVELOPMENT_COMPATIBLE: 2,
}
FORBIDDEN_DECISION_FIELDS = {
    "future_return",
    "oracle_action",
    "realized_pnl",
    "resolved_outcome",
    "settlement_pnl",
    "settlement_return",
    "target_net_return_after_cost",
    "total_net_pnl_per_notional",
    "total_net_return",
}
REQUIRED_FILES = (
    "polymarket_corpus_manifest.json",
    "training_corpus_provenance.json",
    "polymarket_market_metadata.jsonl",
    "polymarket_feature_rows.jsonl",
    "polymarket_token_book_snapshots.jsonl",
    "polymarket_token_trades.jsonl",
    "polymarket_btc_reference_candles.jsonl",
    "polymarket_chainlink_prices.jsonl",
    "polymarket_chainlink_decision_time_evidence_manifest.json",
    "polymarket_label_rows.jsonl",
    "polymarket_resolution_events.jsonl",
)
NON_EMPTY_JSONL_FILES = tuple(
    filename for filename in REQUIRED_FILES if filename.endswith(".jsonl")
)
NORMALIZED_HASH_KEYS = {
    "polymarket_market_metadata.jsonl": "market_metadata",
    "polymarket_feature_rows.jsonl": "feature_rows",
    "polymarket_token_book_snapshots.jsonl": "token_book_snapshots",
    "polymarket_token_trades.jsonl": "token_trades",
    "polymarket_btc_reference_candles.jsonl": "btc_reference_candles",
    "polymarket_chainlink_prices.jsonl": "chainlink_prices",
    "polymarket_chainlink_decision_time_evidence_manifest.json": (
        "chainlink_decision_time_evidence_manifest"
    ),
    "polymarket_label_rows.jsonl": "label_rows",
    "polymarket_resolution_events.jsonl": "resolution_events",
}
BASE_FEATURE_FIELDS = (
    "btc_return_10s",
    "btc_return_30s",
    "btc_return_1m",
    "btc_return_5m",
    "btc_return_15m",
    "btc_volatility_1m",
    "btc_volatility_5m",
    "btc_volatility_15m",
    "reference_price_to_beat_distance_at_decision",
    "time_to_close_seconds",
    "market_age_seconds",
    "combined_spread_bps",
    "liquidity_imbalance",
    "recent_up_trade_volume",
    "recent_down_trade_volume",
)
SIDE_FEATURE_SUFFIXES = (
    "bid",
    "ask",
    "spread_bps",
    "queue_fill_probability_proxy",
    "book_staleness_ms",
    "liquidity_depth",
    "executable_ask_notional",
    "executable_bid_notional",
    "recent_book_update_count_1m",
    "recent_spread_stability_1m",
    "recent_bid_depth_volatility_1m",
)
RAW_REQUIRED_EXACT_KEYS = (
    "raw_polymarket_markets.jsonl",
    "raw_polymarket_orderbooks.jsonl",
    "raw_polymarket_trades.jsonl",
    "raw_polymarket_resolutions.jsonl",
    "raw_polymarket_chainlink_prices.jsonl",
)
CONVERSION_ONLY_REASON_CODES = {
    "current_base_feature_fields_missing",
    "current_execution_compatibility_derived_fields_missing",
    "current_runtime_feature_fields_missing",
}


@dataclass(frozen=True, slots=True)
class HistoricalCorpusCompatibilityAuditConfig:
    """Immutable inputs for a read-only historical corpus audit."""

    run_id: str
    corpus_root: Path | str
    output_dir: Path | str
    protocol_path: Path | str
    feature_contract_path: Path | str
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        object.__setattr__(self, "corpus_root", Path(self.corpus_root))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "protocol_path", Path(self.protocol_path))
        object.__setattr__(self, "feature_contract_path", Path(self.feature_contract_path))


def run_historical_corpus_compatibility_audit(
    config: HistoricalCorpusCompatibilityAuditConfig,
) -> dict[str, Any]:
    """Audit historical corpora without parsing label or resolution values."""

    corpus_root = config.corpus_root.resolve()
    protocol_path = config.protocol_path.resolve()
    feature_contract_path = config.feature_contract_path.resolve()
    if not corpus_root.is_dir():
        raise ValueError(f"corpus root does not exist: {corpus_root}")
    protocol = _load_json(protocol_path)
    validate_pairwise_action_advantage_lcb_protocol(protocol)
    protocol_sha256 = _sha256_file(protocol_path)
    feature_contract = _load_json(feature_contract_path)
    validate_pairwise_action_advantage_lcb_feature_contract(
        feature_contract,
        expected_parent_protocol_sha256=protocol_sha256,
    )

    run_dir = (config.output_dir / config.run_id).resolve()
    if run_dir.exists() and not config.overwrite_existing:
        raise ValueError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_paths = sorted(
        path
        for path in corpus_root.rglob("polymarket_corpus_manifest.json")
        if path.parent.name.startswith("btc-updown-")
    )
    collector_contract = dict(protocol["collector_contract"])
    rows = [
        _audit_corpus(
            corpus_dir=manifest_path.parent,
            collector_contract=collector_contract,
        )
        for manifest_path in manifest_paths
    ]
    _apply_deduplication(rows)

    rows_path = run_dir / "historical_corpus_compatibility_rows.jsonl"
    _write_jsonl(rows_path, rows)
    report = _build_report(
        config=config,
        corpus_root=corpus_root,
        protocol_path=protocol_path,
        feature_contract_path=feature_contract_path,
        rows=rows,
        rows_path=rows_path,
    )
    report_path = run_dir / "historical_corpus_compatibility_audit_report.json"
    _write_json(report_path, report)
    markdown_path = run_dir / "historical_corpus_compatibility_audit_report.md"
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")

    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-manifest-v1",
        "run_id": config.run_id,
        "report": _descriptor(report_path),
        "report_markdown": _descriptor(markdown_path),
        "compatibility_rows": _descriptor(rows_path),
        "protocol": _descriptor(protocol_path),
        "feature_contract": _descriptor(feature_contract_path),
        "discovered_corpus_count": report["discovered_corpus_count"],
        "unique_market_count": report["unique_market_count"],
        "input_inventory_hash": report["input_inventory_hash"],
        "historical_development_compatible_market_count": report[
            "historical_development_compatible_market_count"
        ],
        "fresh_confirmatory_eligible_market_count": 0,
        "outcome_values_loaded": False,
        "pnl_values_loaded": False,
        "active_issue175_collection_mutated": False,
        **compact_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "historical_corpus_compatibility_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "report_path": report_path,
        "markdown_path": markdown_path,
        "rows_path": rows_path,
        "manifest_path": manifest_path,
        "report": report,
        "manifest": manifest,
    }


def _audit_corpus(
    *,
    corpus_dir: Path,
    collector_contract: dict[str, Any],
) -> dict[str, Any]:
    reason_codes: list[str] = []
    file_inventory: dict[str, dict[str, Any] | None] = {}
    for filename in REQUIRED_FILES:
        path = corpus_dir / filename
        file_inventory[filename] = _file_descriptor(path) if path.is_file() else None
        if not path.is_file():
            reason_codes.append(f"required_file_missing:{filename}")
        elif filename in NON_EMPTY_JSONL_FILES and int(path.stat().st_size) == 0:
            reason_codes.append(f"required_stream_empty:{filename}")

    manifest_path = corpus_dir / "polymarket_corpus_manifest.json"
    provenance_path = corpus_dir / "training_corpus_provenance.json"
    manifest = _load_json(manifest_path) if manifest_path.is_file() else {}
    provenance = _load_json(provenance_path) if provenance_path.is_file() else {}
    _audit_manifest_and_provenance(
        manifest_path=manifest_path,
        manifest=manifest,
        provenance=provenance,
        reason_codes=reason_codes,
    )
    _audit_artifact_hashes(
        corpus_dir=corpus_dir,
        manifest=manifest,
        reason_codes=reason_codes,
    )
    raw_hash_summary = _raw_hash_summary(manifest)
    if raw_hash_summary["raw_evidence_hash_complete"] is not True:
        reason_codes.append("raw_evidence_hash_inventory_incomplete")

    safe_rows: dict[str, list[dict[str, Any]]] = {}
    for filename in (
        "polymarket_market_metadata.jsonl",
        "polymarket_feature_rows.jsonl",
        "polymarket_token_book_snapshots.jsonl",
        "polymarket_btc_reference_candles.jsonl",
        "polymarket_chainlink_prices.jsonl",
    ):
        path = corpus_dir / filename
        if not path.is_file() or path.stat().st_size == 0:
            safe_rows[filename] = []
            continue
        try:
            rows = _load_jsonl(path)
        except (json.JSONDecodeError, ValueError):
            reason_codes.append(f"decision_time_stream_invalid_json:{filename}")
            safe_rows[filename] = []
            continue
        forbidden = sorted(
            {
                field
                for row in rows
                for field in _find_fields(row, FORBIDDEN_DECISION_FIELDS)
            }
        )
        if forbidden:
            reason_codes.append(f"forbidden_decision_fields_present:{filename}")
        safe_rows[filename] = rows

    features = safe_rows["polymarket_feature_rows.jsonl"]
    metadata_rows = safe_rows["polymarket_market_metadata.jsonl"]
    books = safe_rows["polymarket_token_book_snapshots.jsonl"]
    chainlink_rows = safe_rows["polymarket_chainlink_prices.jsonl"]
    market_id = _market_id(features, metadata_rows, provenance)
    market_identity_summary = _audit_market_identity(
        features=features,
        metadata_rows=metadata_rows,
        market_id=market_id,
        reason_codes=reason_codes,
    )
    feature_summary = _audit_features(
        features=features,
        collector_contract=collector_contract,
        reason_codes=reason_codes,
    )
    book_summary = _audit_books(
        books=books,
        market_id=market_id,
        feature_count=len(features),
        reason_codes=reason_codes,
    )
    chainlink_summary = _audit_chainlink(
        corpus_dir=corpus_dir,
        manifest=manifest,
        provenance=provenance,
        features=features,
        chainlink_rows=chainlink_rows,
        reason_codes=reason_codes,
    )
    label_summary = _audit_label_contract(
        corpus_dir=corpus_dir,
        manifest=manifest,
        feature_count=len(features),
        reason_codes=reason_codes,
    )

    unique_reasons = sorted(set(reason_codes))
    hard_reasons = [
        reason
        for reason in unique_reasons
        if reason not in CONVERSION_ONLY_REASON_CODES
    ]
    if not unique_reasons:
        classification = HISTORICAL_DEVELOPMENT_COMPATIBLE
    elif (
        not hard_reasons
        and raw_hash_summary["raw_evidence_hash_complete"] is True
        and book_summary["complete_up_down_book_evidence"] is True
        and chainlink_summary["chainlink_contract_passed"] is True
        and label_summary["cost_aware_label_contract_identified"] is True
    ):
        classification = HISTORICAL_DEVELOPMENT_CONVERTIBLE
    else:
        classification = HISTORICAL_INCOMPATIBLE

    row = {
        "schema_version": f"{SCHEMA_PREFIX}-row-v1",
        "corpus_dir": str(corpus_dir.resolve()),
        "corpus_id": str(provenance.get("corpus_id") or corpus_dir.name),
        "market_id": market_id,
        "round_slug": str(provenance.get("round_slug") or corpus_dir.name),
        "classification": classification,
        "historical_development_fit_eligible": (
            classification == HISTORICAL_DEVELOPMENT_COMPATIBLE
        ),
        "historical_development_rebuild_candidate": (
            classification == HISTORICAL_DEVELOPMENT_CONVERTIBLE
        ),
        "fresh_calibration_eligible": False,
        "fresh_confirmatory_eligible": False,
        "current_issue175_role_eligible": False,
        "reason_codes": unique_reasons,
        "conversion_reason_codes": sorted(
            reason for reason in unique_reasons if reason in CONVERSION_ONLY_REASON_CODES
        ),
        "file_inventory": file_inventory,
        "raw_evidence_hash_summary": raw_hash_summary,
        "market_identity_summary": market_identity_summary,
        "feature_summary": feature_summary,
        "book_summary": book_summary,
        "chainlink_summary": chainlink_summary,
        "label_contract_summary": label_summary,
        "deduplication_status": "pending",
        "duplicate_of_corpus_dir": None,
        "label_rows_content_parsed": False,
        "resolution_rows_content_parsed": False,
        "outcome_values_loaded": False,
        "pnl_values_loaded": False,
        **compact_safety_fields(),
    }
    row["row_id"] = canonical_json_sha256(row)
    return row


def _audit_manifest_and_provenance(
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    provenance: dict[str, Any],
    reason_codes: list[str],
) -> None:
    if manifest.get("schema_version") != CURRENT_CORPUS_SCHEMA_VERSION:
        reason_codes.append("current_corpus_schema_not_present")
    safety = (
        manifest.get("paper_only") is True
        and manifest.get("capital_at_risk") is False
        and manifest.get("polymarket_write_enabled") is False
        and manifest.get("wallet_signing_enabled") is False
    )
    if not safety:
        reason_codes.append("corpus_safety_contract_invalid")
    if provenance:
        if provenance.get("real_historical_corpus_used") is not True:
            reason_codes.append("real_historical_corpus_provenance_missing")
        if provenance.get("manual_live_evidence_eligible") is not True:
            reason_codes.append("manual_live_evidence_eligibility_missing")
        if provenance.get("synthetic_corpus_used") is not False:
            reason_codes.append("synthetic_corpus_provenance_detected")
        if provenance.get("synthetic_public_data_used") is not False:
            reason_codes.append("synthetic_public_data_provenance_detected")
        if provenance.get("mock_public_data_used") is not False:
            reason_codes.append("mock_public_data_provenance_detected")
        if provenance.get("round_scoped_export") is not True:
            reason_codes.append("round_scoped_export_provenance_missing")
        manifest_sha256 = str(provenance.get("phase2_corpus_manifest_sha256") or "")
        if not _is_sha256(manifest_sha256):
            reason_codes.append("phase2_corpus_manifest_provenance_hash_missing")
        elif manifest_sha256 != _sha256_file(manifest_path):
            reason_codes.append("phase2_corpus_manifest_provenance_hash_mismatch")


def _audit_artifact_hashes(
    *,
    corpus_dir: Path,
    manifest: dict[str, Any],
    reason_codes: list[str],
) -> None:
    normalized_hashes = dict(manifest.get("normalized_artifact_hashes") or {})
    for filename, hash_key in NORMALIZED_HASH_KEYS.items():
        path = corpus_dir / filename
        if not path.is_file():
            continue
        expected = str(normalized_hashes.get(hash_key) or "")
        if not _is_sha256(expected):
            reason_codes.append(f"normalized_artifact_hash_missing:{hash_key}")
        elif expected != _sha256_file(path):
            reason_codes.append(f"normalized_artifact_hash_mismatch:{hash_key}")


def _raw_hash_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    raw_hashes = dict(manifest.get("raw_artifact_hashes") or {})
    missing = [
        key
        for key in RAW_REQUIRED_EXACT_KEYS
        if not _is_sha256(str(raw_hashes.get(key) or ""))
    ]
    btc_keys = sorted(
        key
        for key, value in raw_hashes.items()
        if ("btc" in key.lower() or "coinbase" in key.lower() or "kraken" in key.lower())
        and _is_sha256(str(value or ""))
    )
    if not btc_keys:
        missing.append("raw_btc_feature_candles")
    return {
        "raw_artifact_hash_count": len(raw_hashes),
        "raw_btc_feature_keys": btc_keys,
        "missing_required_raw_artifact_hashes": sorted(set(missing)),
        "raw_evidence_hash_complete": not missing,
    }


def _audit_features(
    *,
    features: list[dict[str, Any]],
    collector_contract: dict[str, Any],
    reason_codes: list[str],
) -> dict[str, Any]:
    if not features:
        reason_codes.append("feature_rows_empty")
    causality_violations = 0
    missing_base = Counter()
    missing_runtime = Counter()
    execution_missing = Counter()
    execution_blockers = Counter()
    required_runtime = tuple(
        str(value)
        for value in collector_contract.get("required_runtime_feature_fields") or []
    )
    maximum_staleness = min(
        float(collector_contract["maximum_selected_side_book_staleness_ms"]),
        float(collector_contract["maximum_opposite_side_book_staleness_ms"]),
    )
    for row in features:
        decision_ts = int(row.get("decision_ts") or 0)
        max_input_ts = int(row.get("max_input_ts") or 0)
        if decision_ts <= 0 or max_input_ts > decision_ts:
            causality_violations += 1
        values = dict(row.get("features") or {})
        for field in BASE_FEATURE_FIELDS:
            if not _finite(values.get(field)):
                missing_base[field] += 1
        for field in required_runtime:
            if not _finite(values.get(field)):
                missing_runtime[field] += 1
        for side in ("up", "down"):
            bid = _finite_float(values.get(f"{side}_bid"))
            ask = _finite_float(values.get(f"{side}_ask"))
            if bid is None or ask is None:
                execution_missing[f"{side}_executable_book_missing"] += 1
            elif not 0.0 < bid <= ask < 1.0:
                execution_blockers[f"{side}_executable_book_invalid"] += 1
            staleness = _finite_float(values.get(f"{side}_book_staleness_ms"))
            if staleness is None:
                execution_missing[f"{side}_book_staleness_missing"] += 1
            elif staleness < 0.0 or staleness > maximum_staleness:
                execution_blockers[f"{side}_book_staleness_invalid"] += 1
            queue = _finite_float(values.get(f"{side}_queue_fill_probability_proxy"))
            if queue is None:
                execution_missing[f"{side}_queue_fill_missing"] += 1
            elif not 0.0 <= queue <= 1.0:
                execution_blockers[f"{side}_queue_fill_invalid"] += 1
            for suffix in SIDE_FEATURE_SUFFIXES:
                if not _finite(values.get(f"{side}_{suffix}")):
                    missing_base[f"{side}_{suffix}"] += 1
    if causality_violations:
        reason_codes.append("feature_timestamp_causality_violation")
    if missing_base:
        reason_codes.append("current_base_feature_fields_missing")
    if missing_runtime:
        reason_codes.append("current_runtime_feature_fields_missing")
    if execution_missing:
        reason_codes.append("current_execution_compatibility_derived_fields_missing")
    if execution_blockers:
        reason_codes.append("current_execution_compatibility_failed")
    return {
        "feature_row_count": len(features),
        "feature_timestamp_causality_violation_count": causality_violations,
        "missing_base_feature_distribution": dict(sorted(missing_base.items())),
        "missing_runtime_feature_distribution": dict(sorted(missing_runtime.items())),
        "missing_execution_derived_field_distribution": dict(
            sorted(execution_missing.items())
        ),
        "execution_compatibility_blocking_distribution": dict(
            sorted(execution_blockers.items())
        ),
        "all_feature_rows_current_compatible": bool(features)
        and causality_violations == 0
        and not missing_base
        and not missing_runtime
        and not execution_missing
        and not execution_blockers,
    }


def _audit_books(
    *,
    books: list[dict[str, Any]],
    market_id: str,
    feature_count: int,
    reason_codes: list[str],
) -> dict[str, Any]:
    outcomes = {str(row.get("outcome") or "") for row in books}
    invalid_rows = 0
    market_mismatch_count = 0
    for row in books:
        if market_id and str(row.get("market_id") or "") != market_id:
            market_mismatch_count += 1
        bid = _finite_float(row.get("bid_price"))
        ask = _finite_float(row.get("ask_price"))
        ts = int(row.get("ts") or 0)
        available_at_ts = int(row.get("available_at_ts") or 0)
        if (
            bid is None
            or ask is None
            or not 0.0 < bid <= ask < 1.0
            or ts <= 0
            or available_at_ts < ts
        ):
            invalid_rows += 1
    if outcomes != {"UP", "DOWN"}:
        reason_codes.append("complete_up_down_book_evidence_missing")
    if feature_count > 0 and len(books) < feature_count * 2:
        reason_codes.append("book_snapshot_pair_coverage_incomplete")
    if invalid_rows:
        reason_codes.append("book_snapshot_rows_invalid")
    if market_mismatch_count:
        reason_codes.append("book_market_identity_mismatch")
    return {
        "book_row_count": len(books),
        "book_outcomes": sorted(outcomes),
        "invalid_book_row_count": invalid_rows,
        "book_market_identity_mismatch_count": market_mismatch_count,
        "minimum_expected_book_row_count": feature_count * 2,
        "complete_up_down_book_evidence": outcomes == {"UP", "DOWN"}
        and invalid_rows == 0
        and market_mismatch_count == 0
        and len(books) >= feature_count * 2,
    }


def _audit_chainlink(
    *,
    corpus_dir: Path,
    manifest: dict[str, Any],
    provenance: dict[str, Any],
    features: list[dict[str, Any]],
    chainlink_rows: list[dict[str, Any]],
    reason_codes: list[str],
) -> dict[str, Any]:
    chainlink_manifest_path = (
        corpus_dir / "polymarket_chainlink_decision_time_evidence_manifest.json"
    )
    chainlink_manifest = (
        _load_json(chainlink_manifest_path) if chainlink_manifest_path.is_file() else {}
    )
    feature_reference_missing = 0
    feature_reference_provenance_invalid = 0
    feature_reference_causality_violations = 0
    for row in features:
        values = dict(row.get("features") or {})
        if not _finite(values.get("reference_price_to_beat_distance_at_decision")):
            feature_reference_missing += 1
        reference_provenance = (
            (row.get("feature_provenance") or {}).get(
                "reference_price_to_beat_distance_at_decision"
            )
            or {}
        )
        if (
            reference_provenance.get("reference_price_to_beat_source")
            != "polymarket_rtds_chainlink_market_start"
            or reference_provenance.get("provenance_valid") is not True
        ):
            feature_reference_provenance_invalid += 1
        decision_ts = int(row.get("decision_ts") or 0)
        if (
            int(reference_provenance.get("max_input_ts") or 0) > decision_ts
            or int(reference_provenance.get("available_at_ts") or 0) > decision_ts
        ):
            feature_reference_causality_violations += 1
    integration = dict(manifest.get("chainlink_decision_time_feature_integration") or {})
    provenance_integration = dict(
        provenance.get("chainlink_decision_time_evidence") or {}
    )
    chainlink_path = corpus_dir / "polymarket_chainlink_prices.jsonl"
    chainlink_evidence_sha256 = (
        _sha256_file(chainlink_path) if chainlink_path.is_file() else ""
    )
    chainlink_manifest_sha256 = (
        _sha256_file(chainlink_manifest_path)
        if chainlink_manifest_path.is_file()
        else ""
    )
    contract_passed = (
        bool(chainlink_rows)
        and chainlink_manifest.get("source_type") == "polymarket_rtds_chainlink"
        and chainlink_manifest.get("decision_time_only") is True
        and chainlink_manifest.get("feature_builder_integration_passed") is True
        and chainlink_manifest.get("feature_builder_integration_required") is False
        and int(chainlink_manifest.get("timestamp_causality_violation_count") or 0) == 0
        and int(chainlink_manifest.get("integrated_feature_row_count") or 0)
        == len(features)
        and int(chainlink_manifest.get("missing_or_invalid_feature_row_count") or 0)
        == 0
        and int(chainlink_manifest.get("row_count") or 0) == len(chainlink_rows)
        and chainlink_manifest.get("evidence_sha256") == chainlink_evidence_sha256
        and integration == chainlink_manifest
        and provenance_integration.get("attached") is True
        and provenance_integration.get("feature_builder_integration_passed") is True
        and provenance_integration.get("evidence_sha256") == chainlink_evidence_sha256
        and provenance_integration.get("manifest_sha256") == chainlink_manifest_sha256
        and feature_reference_missing == 0
        and feature_reference_provenance_invalid == 0
        and feature_reference_causality_violations == 0
    )
    if not contract_passed:
        reason_codes.append("chainlink_decision_time_contract_failed")
    return {
        "chainlink_row_count": len(chainlink_rows),
        "chainlink_source_type": chainlink_manifest.get("source_type"),
        "integrated_feature_row_count": int(
            chainlink_manifest.get("integrated_feature_row_count") or 0
        ),
        "feature_reference_missing_count": feature_reference_missing,
        "feature_reference_provenance_invalid_count": (
            feature_reference_provenance_invalid
        ),
        "feature_reference_causality_violation_count": (
            feature_reference_causality_violations
        ),
        "chainlink_contract_passed": contract_passed,
    }


def _audit_label_contract(
    *,
    corpus_dir: Path,
    manifest: dict[str, Any],
    feature_count: int,
    reason_codes: list[str],
) -> dict[str, Any]:
    label_path = corpus_dir / "polymarket_label_rows.jsonl"
    resolution_path = corpus_dir / "polymarket_resolution_events.jsonl"
    label_row_count = _line_count(label_path) if label_path.is_file() else 0
    resolution_row_count = _line_count(resolution_path) if resolution_path.is_file() else 0
    expected_label_count = feature_count * 5
    identified = (
        manifest.get("schema_version") == CURRENT_CORPUS_SCHEMA_VERSION
        and manifest.get("sell_before_close_label_schema_version")
        == CURRENT_SBC_LABEL_SCHEMA_VERSION
        and manifest.get("sell_before_close_label_gate_passed") is True
        and int(manifest.get("label_row_count") or 0) == label_row_count
        and label_row_count == expected_label_count
        and resolution_row_count > 0
    )
    if label_row_count != expected_label_count:
        reason_codes.append("complete_five_action_label_count_missing")
    if not identified:
        reason_codes.append("current_cost_aware_label_contract_not_identified")
    return {
        "label_row_count": label_row_count,
        "expected_five_action_label_row_count": expected_label_count,
        "resolution_row_count": resolution_row_count,
        "corpus_schema_version": manifest.get("schema_version"),
        "sell_before_close_label_schema_version": manifest.get(
            "sell_before_close_label_schema_version"
        ),
        "sell_before_close_label_gate_passed": manifest.get(
            "sell_before_close_label_gate_passed"
        ),
        "cost_aware_label_contract_identified": identified,
        "label_rows_content_parsed": False,
        "resolution_rows_content_parsed": False,
    }


def _apply_deduplication(rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        identity = str(row.get("market_id") or "")
        if not identity:
            identity = f"missing-market:{row['corpus_id']}:{row['corpus_dir']}"
        grouped[identity].append(row)
    for group in grouped.values():
        selected = sorted(
            group,
            key=lambda row: (
                -CLASSIFICATION_PRIORITY[str(row["classification"])],
                len(str(row["corpus_dir"]).split("/")),
                str(row["corpus_dir"]),
            ),
        )[0]
        selected["deduplication_status"] = "selected_unique_market"
        selected["duplicate_of_corpus_dir"] = None
        selected["row_id"] = canonical_json_sha256(
            {key: value for key, value in selected.items() if key != "row_id"}
        )
        for row in group:
            if row is selected:
                continue
            row["deduplication_status"] = "duplicate_excluded"
            row["duplicate_of_corpus_dir"] = selected["corpus_dir"]
            row["row_id"] = canonical_json_sha256(
                {key: value for key, value in row.items() if key != "row_id"}
            )


def _build_report(
    *,
    config: HistoricalCorpusCompatibilityAuditConfig,
    corpus_root: Path,
    protocol_path: Path,
    feature_contract_path: Path,
    rows: list[dict[str, Any]],
    rows_path: Path,
) -> dict[str, Any]:
    unique_rows = [
        row for row in rows if row["deduplication_status"] == "selected_unique_market"
    ]
    classification_distribution = Counter(
        str(row["classification"]) for row in unique_rows
    )
    reason_distribution = Counter(
        reason for row in unique_rows for reason in row["reason_codes"]
    )
    input_inventory = [
        {
            "corpus_dir": row["corpus_dir"],
            "corpus_manifest_sha256": (
                (row["file_inventory"].get("polymarket_corpus_manifest.json") or {}).get(
                    "sha256"
                )
            ),
            "training_corpus_provenance_sha256": (
                (row["file_inventory"].get("training_corpus_provenance.json") or {}).get(
                    "sha256"
                )
            ),
        }
        for row in rows
    ]
    input_inventory_hash = canonical_json_sha256(input_inventory)
    compatible = int(
        classification_distribution[HISTORICAL_DEVELOPMENT_COMPATIBLE]
    )
    convertible = int(
        classification_distribution[HISTORICAL_DEVELOPMENT_CONVERTIBLE]
    )
    reusable_train = min(90, compatible)
    minimum_fresh_valid = (90 - reusable_train) + 45 + 60
    full_fresh_valid = 195
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-report-v1",
        "run_id": config.run_id,
        "audit_mode": "outcome_blind_read_only_historical_compatibility",
        "corpus_root": str(corpus_root),
        "protocol": _descriptor(protocol_path),
        "feature_contract": _descriptor(feature_contract_path),
        "compatibility_rows": _descriptor(rows_path),
        "input_inventory_hash": input_inventory_hash,
        "input_inventory_entry_count": len(input_inventory),
        "discovered_corpus_count": len(rows),
        "unique_market_count": len(unique_rows),
        "duplicate_excluded_corpus_count": len(rows) - len(unique_rows),
        "classification_distribution": dict(sorted(classification_distribution.items())),
        "historical_development_compatible_market_count": compatible,
        "historical_development_convertible_market_count": convertible,
        "historical_incompatible_market_count": int(
            classification_distribution[HISTORICAL_INCOMPATIBLE]
        ),
        "fresh_calibration_eligible_market_count": 0,
        "fresh_confirmatory_eligible_market_count": 0,
        "compatibility_reason_distribution": dict(sorted(reason_distribution.items())),
        "outcome_blind_access_audit": {
            "label_rows_content_parsed": False,
            "resolution_rows_content_parsed": False,
            "outcome_values_loaded": False,
            "pnl_values_loaded": False,
            "oracle_values_loaded": False,
            "validation_metrics_loaded": False,
            "byte_level_hash_and_row_count_only_for_label_and_resolution_files": True,
            "classification_uses_only": [
                "corpus_and_provenance_manifests",
                "decision_time_feature_rows",
                "decision_time_market_metadata",
                "decision_time_book_snapshots",
                "decision_time_btc_candles",
                "decision_time_chainlink_rows_and_provenance",
                "artifact_hashes_and_non_empty_stream_counts",
            ],
        },
        "future_hybrid_protocol_planning_estimate": {
            "planning_only": True,
            "current_issue175_protocol_changed": False,
            "current_issue175_capture_target_remains": 210,
            "current_issue175_valid_market_target_remains": 195,
            "historical_development_train_target": 90,
            "historical_development_train_reusable_market_count": reusable_train,
            "fresh_development_train_market_count_still_required": 90 - reusable_train,
            "fresh_calibration_market_count_still_required": 45,
            "fresh_confirmatory_market_count_still_required": 60,
            "minimum_fresh_valid_market_count_for_future_hybrid": minimum_fresh_valid,
            "quality_buffer_attempt_count_if_preserved": 15,
            "estimated_future_hybrid_capture_attempt_count": minimum_fresh_valid + 15,
            "estimated_rounds_saved_vs_210_attempt_protocol": (
                full_fresh_valid + 15 - (minimum_fresh_valid + 15)
            ),
            "estimated_wall_clock_hours_saved_at_5m_per_round": (
                reusable_train * 5.0 / 60.0
            ),
            "fresh_confirmatory_history_substitution_allowed": False,
            "new_issue_branch_and_precollection_freeze_required": True,
        },
        "active_collection_invariants": {
            "issue175_collection_mutated": False,
            "issue178_collector_mutated": False,
            "issue179_support_gate_mutated": False,
            "current_collection_may_not_be_reclassified_mid_run": True,
        },
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "v8_execution_handoff_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        **compact_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    return report


def _render_markdown(report: dict[str, Any]) -> str:
    distribution = report["classification_distribution"]
    reasons = report["compatibility_reason_distribution"]
    planning = report["future_hybrid_protocol_planning_estimate"]
    lines = [
        "# Historical Corpus Compatibility Audit",
        "",
        f"- run id: `{report['run_id']}`",
        f"- discovered corpora: `{report['discovered_corpus_count']}`",
        f"- unique markets: `{report['unique_market_count']}`",
        f"- duplicate corpora excluded: `{report['duplicate_excluded_corpus_count']}`",
        (
            "- historical development compatible: "
            f"`{report['historical_development_compatible_market_count']}`"
        ),
        (
            "- historical development convertible: "
            f"`{report['historical_development_convertible_market_count']}`"
        ),
        f"- historical incompatible: `{report['historical_incompatible_market_count']}`",
        "- fresh confirmatory eligible: `0`",
        "- outcome/PnL values loaded: `false`",
        "- current #175 protocol changed: `false`",
        "",
        "## Classification",
        "",
        "| classification | unique markets |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| `{name}` | {count} |" for name, count in sorted(distribution.items())
    )
    lines.extend(
        [
            "",
            "## Primary Exclusion Reasons",
            "",
            "| reason | markets |",
            "| --- | ---: |",
        ]
    )
    lines.extend(
        f"| `{name}` | {count} |"
        for name, count in sorted(reasons.items(), key=lambda item: (-item[1], item[0]))[
            :30
        ]
    )
    lines.extend(
        [
            "",
            "## Future Hybrid Planning Estimate",
            "",
            (
                "- reusable historical development markets: "
                f"`{planning['historical_development_train_reusable_market_count']}`"
            ),
            (
                "- minimum fresh valid markets in a new hybrid protocol: "
                f"`{planning['minimum_fresh_valid_market_count_for_future_hybrid']}`"
            ),
            (
                "- estimated attempts with the existing 15-round quality buffer: "
                f"`{planning['estimated_future_hybrid_capture_attempt_count']}`"
            ),
            (
                "- estimated 5m wall-clock saving: "
                f"`{planning['estimated_wall_clock_hours_saved_at_5m_per_round']:.2f}h`"
            ),
            "",
            "This estimate is planning evidence only. It does not change #175 and cannot "
            "replace fresh calibration or confirmatory markets.",
            "",
            "## Safety",
            "",
            "- paper only: `true`",
            "- capital at risk: `false`",
            "- Polymarket writes: `false`",
            "- wallet signing: `false`",
            "- source/freeze/promotion/handoff: `false`",
        ]
    )
    return "\n".join(lines) + "\n"


def _market_id(
    features: list[dict[str, Any]],
    metadata_rows: list[dict[str, Any]],
    provenance: dict[str, Any],
) -> str:
    candidates = {
        str(row.get("market_id") or "")
        for row in [*features, *metadata_rows]
        if str(row.get("market_id") or "")
    }
    if len(candidates) == 1:
        return next(iter(candidates))
    corpus_id = str(provenance.get("corpus_id") or "")
    return corpus_id if corpus_id.startswith("0x") else ""


def _audit_market_identity(
    *,
    features: list[dict[str, Any]],
    metadata_rows: list[dict[str, Any]],
    market_id: str,
    reason_codes: list[str],
) -> dict[str, Any]:
    feature_market_ids = sorted(
        {
            str(row.get("market_id") or "")
            for row in features
            if str(row.get("market_id") or "")
        }
    )
    metadata_market_ids = sorted(
        {
            str(row.get("market_id") or "")
            for row in metadata_rows
            if str(row.get("market_id") or "")
        }
    )
    consistent = (
        bool(market_id)
        and feature_market_ids == [market_id]
        and metadata_market_ids == [market_id]
    )
    if not consistent:
        reason_codes.append("market_identity_missing_or_inconsistent")
    return {
        "market_id": market_id,
        "feature_market_ids": feature_market_ids,
        "metadata_market_ids": metadata_market_ids,
        "market_identity_consistent": consistent,
    }


def _file_descriptor(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
        "row_count": _line_count(path) if path.suffix == ".jsonl" else None,
    }


def _descriptor(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256_file(path)}


def _line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"expected JSON object row: {path}")
        rows.append(payload)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _finite(value: Any) -> bool:
    return _finite_float(value) is not None


def _finite_float(value: Any) -> float | None:
    if not isinstance(value, int | float):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _find_fields(payload: Any, forbidden: set[str]) -> set[str]:
    found: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            if str(key) in forbidden:
                found.add(str(key))
            found.update(_find_fields(value, forbidden))
    elif isinstance(payload, list):
        for value in payload:
            found.update(_find_fields(value, forbidden))
    return found
