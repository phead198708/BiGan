"""Executable authorization and exact-window boundaries for BTC 15m MoE v2."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from bigan.v8.polymarket.challenge_development_lane import sha256_file
from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.moe_confirmatory_v2 import (
    _parse_pytest_junit,
    _parse_ruff_json,
)
from bigan.v8.polymarket.moe_precollection_hardening_r1 import (
    BASE_COMMIT,
    CANDIDATE_BUNDLE_HASH,
    SAFETY,
)
from bigan.v8.polymarket.regime_adaptive_lineage import REPO_ROOT

LINEAGE_ID = "BTC-15M-MoE-confirmatory-v2"
REVISION_ID = "BTC-15M-MoE-confirmatory-v2-collection-boundary-r2"
TARGET_MARKET_COUNT = 800
ATTEMPT_CAP = 926
MARKET_FAMILY = "btc_updown_15m"
MARKET_HORIZON_SECONDS = 900
CREATED_AT = "2026-07-30T14:00:00+00:00"
REQUIRED_RAW_STREAMS = (
    "raw_polymarket_markets.jsonl",
    "raw_polymarket_orderbooks.jsonl",
    "raw_polymarket_trades.jsonl",
    "raw_binance_btcusdt_klines.jsonl",
    "raw_polymarket_chainlink_prices.jsonl",
)
QUALITY_OBSERVATION_FIELDS = (
    "market_identity_complete",
    "provider_capture_complete",
    "paired_executable_asks_complete",
    "book_capture_complete",
    "chainlink_capture_complete",
)
FORBIDDEN_ATTEMPT_FIELDS = {
    "quality_valid",
    "router_route",
    "expert_id",
    "model_prediction",
    "candidate_prediction",
    "baseline_prediction",
    "selection_score",
    "selected_side",
    "accepted",
    "settlement_outcome",
    "outcome",
    "label",
    "resolved_outcome",
    "settlement_result",
    "settlement_price",
    "target",
    "realized_pnl",
    "unit_pnl",
    "net_pnl",
    "gross_pnl",
    "future_price",
    "future_return",
    "post_close_price",
}
FORBIDDEN_NORMALIZED_FIELDS = FORBIDDEN_ATTEMPT_FIELDS - {"quality_valid"}
FORBIDDEN_DECISION_OUTCOME_FIELDS = {
    "settlement_outcome",
    "outcome",
    "label",
    "resolved_outcome",
    "settlement_result",
    "settlement_price",
    "target",
    "realized_pnl",
    "unit_pnl",
    "net_pnl",
    "gross_pnl",
    "future_price",
    "future_return",
    "post_close_price",
}
STATE_BLOCKED = {
    "fresh_collection_authorized": False,
    "fresh_collection_started": False,
    "fresh_outcomes_opened": False,
}
AUTHORIZATION_FROZEN_KEYS = (
    "candidate_artifact_graph",
    "matched_baseline_contract",
    "matched_baseline_artifact",
    "collector_protocol",
    "reporting_contract",
    "statistical_protocol",
    "health_snapshot",
    "health_manifest",
    "attempt_cap_analysis",
    "runtime_validation_report",
)
CAPTURE_MANIFEST_REQUIRED_FIELDS = {
    "schema_version",
    "lineage_id",
    "authorization_artifact_id",
    "authorization_artifact_sha256",
    "collector_protocol_sha256",
    "exact_market_count",
    "ordered_market_ids",
    "ordered_market_ids_sha256",
    "ordered_attempt_indices",
    "ordered_attempt_indices_sha256",
    "first_market_start_ts",
    "last_market_start_ts",
    "strictly_later_authorization_boundary",
    "attempts_consumed",
    "unused_attempt_capacity",
    "cap_exhausted",
    "all_decisions_frozen",
    "candidate_decision_row_count",
    "candidate_decision_rows_sha256",
    "baseline_decision_row_count",
    "baseline_decision_rows_sha256",
    "paired_decision_population_sha256",
    "raw_evidence_manifest_count",
    "raw_evidence_manifest_set_sha256",
    "raw_evidence_manifest_index_sha256",
    "normalized_attempt_ledger_sha256",
    "no_outcome_access_confirmation",
    "no_settlement_confirmation",
    "window_stop_attempt_index",
    "audit_only_post_boundary_attempt_indices",
    "validation_fixture_only",
    "promotion_evidence_eligible",
    "state",
    "safety",
}


@dataclass(frozen=True, slots=True)
class AuthorizationExpectation:
    """Caller-pinned facts required to validate one manual authorization."""

    artifact_path: Path
    expected_artifact_sha256: str
    authorization_request_text_sha256: str
    authorization_decision_text_sha256: str
    approver_identity: str
    authorization_source_id: str
    authorization_source_url: str
    authorization_timestamp: str
    strictly_later_than_timestamp: str
    expected_frozen_artifact_sha256: Mapping[str, str]
    allow_test_fixture: bool = False


def frozen_authorization_artifacts(
    repository_root: Path | str | None = None,
) -> dict[str, dict[str, str]]:
    """Return and verify every repository artifact a manual grant must pin."""

    repo_root = Path(repository_root or REPO_ROOT).resolve()
    v2_dir = repo_root / "examples" / "v8" / "polymarket_configs" / LINEAGE_ID
    baseline_contract = _load_verified_json(v2_dir / "moe_matched_global_baseline_contract.json")
    paths = {
        "candidate_artifact_graph": v2_dir / "moe_artifact_graph.json",
        "matched_baseline_contract": (v2_dir / "moe_matched_global_baseline_contract.json"),
        "matched_baseline_artifact": (repo_root / baseline_contract["artifact"]["path"]),
        "collector_protocol": (v2_dir / "moe_confirmatory_collector_protocol_r2.json"),
        "reporting_contract": (v2_dir / "moe_future_evaluation_reporting_contract.json"),
        "statistical_protocol": (v2_dir / "moe_confirmatory_protocol_r1.json"),
        "health_snapshot": (v2_dir / "collection_attempt_health_snapshot.jsonl"),
        "health_manifest": (v2_dir / "collection_attempt_health_manifest.json"),
        "attempt_cap_analysis": (v2_dir / "collection_quality_rate_analysis_r1.json"),
        "runtime_validation_report": (v2_dir / "moe_artifact_runtime_validation_report.json"),
    }
    artifacts = {key: _descriptor(path, repository_root=repo_root) for key, path in paths.items()}
    if tuple(artifacts) != AUTHORIZATION_FROZEN_KEYS:
        raise ValueError("authorization frozen artifact key set drift")
    graph = _load_verified_json(paths["candidate_artifact_graph"])
    if graph["bundle_hash"] != CANDIDATE_BUNDLE_HASH:
        raise ValueError("candidate bundle hash changed")
    if artifacts["matched_baseline_artifact"]["sha256"] != baseline_contract["artifact"]["sha256"]:
        raise ValueError("matched baseline artifact hash changed")
    return artifacts


def build_manual_authorization_payload(
    *,
    repository_root: Path | str,
    authorization_artifact_id: str,
    authorization_request_text_sha256: str,
    authorization_decision_text_sha256: str,
    approver_identity: str,
    authorization_source_id: str,
    authorization_source_url: str,
    authorization_timestamp: str,
    strictly_later_than_timestamp: str,
    test_fixture_only: bool = False,
) -> dict[str, Any]:
    """Build an in-memory authorization payload; writing it grants nothing."""

    artifacts = frozen_authorization_artifacts(repository_root)
    return {
        "schema_version": "bigan-btc-15m-moe-manual-authorization-r2",
        "lineage_id": LINEAGE_ID,
        "authorization_artifact_id": authorization_artifact_id,
        "authorization_request_text_sha256": (authorization_request_text_sha256),
        "authorization_decision_text_sha256": (authorization_decision_text_sha256),
        "approver_identity": approver_identity,
        "authorization_source_id": authorization_source_id,
        "authorization_source_url": authorization_source_url,
        "authorization_timestamp": authorization_timestamp,
        "strictly_later_than_timestamp": strictly_later_than_timestamp,
        "maximum_attempts": ATTEMPT_CAP,
        "exact_market_target": TARGET_MARKET_COUNT,
        "candidate_bundle_hash": CANDIDATE_BUNDLE_HASH,
        "frozen_artifacts": artifacts,
        "authorization_scope": "outcome_blind_capture_only",
        "explicit_manual_authorization": not test_fixture_only,
        "outcome_blind_capture_authorized": not test_fixture_only,
        "test_fixture_only": test_fixture_only,
        "permissions": {
            "outcome_access_enabled": False,
            "settlement_enabled": False,
            "training_enabled": False,
            "paper_enabled": False,
            "promotion_enabled": False,
            "live_trading_enabled": False,
            "wallet_signing_enabled": False,
            "polymarket_write_enabled": False,
        },
        "state": {
            **STATE_BLOCKED,
            "fresh_collection_authorized": not test_fixture_only,
        },
        "safety": dict(SAFETY),
    }


def validate_manual_collection_authorization(
    expectation: AuthorizationExpectation,
    *,
    repository_root: Path | str | None = None,
) -> dict[str, Any]:
    """Validate one future manual artifact and grant capture-only scope."""

    repo_root = Path(repository_root or REPO_ROOT).resolve()
    artifact_path = expectation.artifact_path.resolve()
    if sha256_file(artifact_path) != expectation.expected_artifact_sha256:
        raise ValueError("authorization artifact SHA-256 mismatch")
    payload = _load_json_object(artifact_path)
    required = {
        "schema_version",
        "lineage_id",
        "authorization_artifact_id",
        "authorization_request_text_sha256",
        "authorization_decision_text_sha256",
        "approver_identity",
        "authorization_source_id",
        "authorization_source_url",
        "authorization_timestamp",
        "strictly_later_than_timestamp",
        "maximum_attempts",
        "exact_market_target",
        "candidate_bundle_hash",
        "frozen_artifacts",
        "authorization_scope",
        "explicit_manual_authorization",
        "outcome_blind_capture_authorized",
        "test_fixture_only",
        "permissions",
        "state",
        "safety",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError("authorization missing required fields: " + ",".join(missing))
    if payload["lineage_id"] != LINEAGE_ID:
        raise ValueError("authorization lineage mismatch")
    if payload["schema_version"] != "bigan-btc-15m-moe-manual-authorization-r2":
        raise ValueError("authorization schema mismatch")
    if not str(payload["authorization_artifact_id"]).strip():
        raise ValueError("authorization artifact ID is required")
    exact_matches = {
        "authorization_request_text_sha256": (expectation.authorization_request_text_sha256),
        "authorization_decision_text_sha256": (expectation.authorization_decision_text_sha256),
        "approver_identity": expectation.approver_identity,
        "authorization_source_id": expectation.authorization_source_id,
        "authorization_source_url": expectation.authorization_source_url,
        "authorization_timestamp": expectation.authorization_timestamp,
        "strictly_later_than_timestamp": (expectation.strictly_later_than_timestamp),
        "maximum_attempts": ATTEMPT_CAP,
        "exact_market_target": TARGET_MARKET_COUNT,
        "candidate_bundle_hash": CANDIDATE_BUNDLE_HASH,
    }
    for field, expected in exact_matches.items():
        actual = payload.get(field)
        if actual != expected:
            raise ValueError(f"authorization field mismatch: {field}")
        if isinstance(expected, str) and not expected.strip():
            raise ValueError(f"authorization field is empty: {field}")
    for field in (
        "authorization_request_text_sha256",
        "authorization_decision_text_sha256",
    ):
        _require_sha256(str(payload[field]), field)
    parsed_url = urlparse(str(payload["authorization_source_url"]))
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("authorization source URL must be absolute HTTP(S)")
    authorization_ts = _parse_timestamp_ms(str(payload["authorization_timestamp"]))
    boundary_ts = _parse_timestamp_ms(str(payload["strictly_later_than_timestamp"]))
    if authorization_ts > boundary_ts:
        raise ValueError("authorization timestamp exceeds collection boundary")

    fixture_only = payload["test_fixture_only"] is True
    if fixture_only and not expectation.allow_test_fixture:
        raise ValueError("test fixture cannot authorize real collection")
    if not fixture_only:
        if payload["explicit_manual_authorization"] is not True:
            raise ValueError("explicit manual authorization is required")
        if payload["outcome_blind_capture_authorized"] is not True:
            raise ValueError("outcome-blind capture authorization is required")
    if payload["authorization_scope"] != "outcome_blind_capture_only":
        raise ValueError("authorization scope is not capture-only")
    permissions = payload["permissions"]
    if not isinstance(permissions, dict) or set(permissions) != {
        "outcome_access_enabled",
        "settlement_enabled",
        "training_enabled",
        "paper_enabled",
        "promotion_enabled",
        "live_trading_enabled",
        "wallet_signing_enabled",
        "polymarket_write_enabled",
    }:
        raise ValueError("authorization permission field set mismatch")
    if any(value is not False for value in permissions.values()):
        raise ValueError("authorization enables a forbidden permission")
    expected_state = {
        **STATE_BLOCKED,
        "fresh_collection_authorized": not fixture_only,
    }
    if payload["state"] != expected_state:
        raise ValueError("authorization state does not match its capture scope")
    if payload["safety"] != SAFETY or any(payload["safety"].values()):
        raise ValueError("authorization safety flags must remain false")

    actual_artifacts = frozen_authorization_artifacts(repo_root)
    expected_hashes = dict(expectation.expected_frozen_artifact_sha256)
    if set(expected_hashes) != set(AUTHORIZATION_FROZEN_KEYS):
        raise ValueError("caller did not pin every frozen authorization hash")
    if set(payload["frozen_artifacts"]) != set(AUTHORIZATION_FROZEN_KEYS):
        raise ValueError("authorization did not pin every frozen artifact")
    for key in AUTHORIZATION_FROZEN_KEYS:
        descriptor = payload["frozen_artifacts"][key]
        actual = actual_artifacts[key]
        if descriptor != actual:
            raise ValueError(f"authorization frozen artifact mismatch: {key}")
        if expected_hashes[key] != actual["sha256"]:
            raise ValueError(f"caller frozen protocol hash mismatch: {key}")
        _verify_descriptor(descriptor, repository_root=repo_root)
    return {
        "authorization_validated": True,
        "capture_only_authorized": not fixture_only,
        "test_fixture_only": fixture_only,
        "authorization_artifact_id": payload["authorization_artifact_id"],
        "authorization_artifact_sha256": expectation.expected_artifact_sha256,
        "authorization_boundary_timestamp": payload["strictly_later_than_timestamp"],
        "authorization_boundary_timestamp_ms": boundary_ts,
        "maximum_attempts": ATTEMPT_CAP,
        "exact_market_target": TARGET_MARKET_COUNT,
        "frozen_artifacts": actual_artifacts,
        "permissions": dict(payload["permissions"]),
        "safety": dict(SAFETY),
    }


def validate_authorized_attempt_ledger(
    *,
    authorization: AuthorizationExpectation,
    attempt_ledger_path: Path | str,
    expected_attempt_ledger_sha256: str,
    normalized_output_path: Path | str,
    repository_root: Path | str | None = None,
) -> dict[str, Any]:
    """Validate all attempts and freeze deterministic normalized rows."""

    repo_root = Path(repository_root or REPO_ROOT).resolve()
    authorization_result = validate_manual_collection_authorization(
        authorization,
        repository_root=repo_root,
    )
    ledger_path = Path(attempt_ledger_path).resolve()
    if sha256_file(ledger_path) != expected_attempt_ledger_sha256:
        raise ValueError("authorized attempt ledger SHA-256 mismatch")
    source_rows = _load_jsonl(ledger_path)
    if len(source_rows) > ATTEMPT_CAP:
        raise ValueError("attempt 927 is forbidden")
    if not source_rows:
        raise ValueError("authorized attempt ledger is empty")
    _validate_source_attempt_hash_chain(source_rows)
    attempt_ids = [str(row.get("attempt_id") or "") for row in source_rows]
    if any(not value for value in attempt_ids):
        raise ValueError("attempt ID is required")
    if len(attempt_ids) != len(set(attempt_ids)):
        raise ValueError("duplicate attempt ID")

    evidence_cache: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    stream_hash_cache: dict[tuple[str, str], bool] = {}
    normalized_rows = []
    boundary_ms = int(authorization_result["authorization_boundary_timestamp_ms"])
    for expected_index, source in enumerate(source_rows, start=1):
        forbidden = _find_forbidden_keys(source, FORBIDDEN_ATTEMPT_FIELDS)
        if forbidden:
            raise ValueError(
                "caller-supplied model, outcome, or quality field: " + ",".join(forbidden)
            )
        if int(source["attempt_index"]) != expected_index:
            raise ValueError("attempt indices must be contiguous from 1")
        if expected_index > ATTEMPT_CAP:
            raise ValueError("attempt 927 is forbidden")
        if source.get("market_family") != MARKET_FAMILY:
            raise ValueError("market family must be btc_updown_15m")
        if int(source.get("market_horizon_seconds") or 0) != MARKET_HORIZON_SECONDS:
            raise ValueError("market horizon must equal 900 seconds")
        market_start_ts = int(source["market_start_ts"])
        if market_start_ts <= boundary_ms:
            raise ValueError("market start must be strictly later than authorization")
        if source.get("paired_executable_ask_capture_attempted") is not True:
            raise ValueError("paired executable ask capture was not attempted")
        if source.get("missing_values_encoded_as_numeric_zero") is not False:
            raise ValueError("missing values may not be encoded as numeric zero")
        if source.get("complement_quote_proxy_used") is not False:
            raise ValueError("complement quote proxy is forbidden")
        available_at_ts = int(source["available_at_ts"])
        decision_ts = int(source["decision_ts"])
        max_input_ts = int(source["max_input_ts"])
        feature_cutoff_ts = int(source["feature_cutoff_ts"])
        if not (
            available_at_ts <= decision_ts
            and max_input_ts <= decision_ts
            and feature_cutoff_ts <= decision_ts
        ):
            raise ValueError("attempt causality violation")
        evidence = _load_raw_evidence_record(
            source,
            repository_root=repo_root,
            index_cache=evidence_cache,
            stream_hash_cache=stream_hash_cache,
        )
        if evidence["attempt_id"] != source["attempt_id"]:
            raise ValueError("raw evidence attempt ID mismatch")
        if evidence["market_id"] != source["market_id"]:
            raise ValueError("raw evidence market ID mismatch")
        if evidence["paired_executable_ask_capture_attempted"] is not True:
            raise ValueError("raw evidence did not attempt paired executable asks")
        evidence_causality = evidence["causality"]
        if (
            int(evidence_causality["available_at_ts"]) != available_at_ts
            or int(evidence_causality["decision_ts"]) != decision_ts
            or int(evidence_causality["max_input_ts"]) != max_input_ts
            or int(evidence_causality["feature_cutoff_ts"]) != feature_cutoff_ts
        ):
            raise ValueError("raw evidence causality does not match attempt")
        if evidence["missing_values_encoded_as_numeric_zero"] is not False:
            raise ValueError("raw evidence encodes missing as numeric zero")
        if evidence["complement_quote_proxy_used"] is not False:
            raise ValueError("raw evidence uses complement quote proxy")
        quality = _derive_quality_from_evidence(evidence)
        normalized_rows.append(
            {
                "schema_version": "bigan-btc-15m-authorized-attempt-r2",
                "lineage_id": LINEAGE_ID,
                "authorization_artifact_id": authorization_result["authorization_artifact_id"],
                "authorization_artifact_sha256": authorization_result[
                    "authorization_artifact_sha256"
                ],
                "authorization_boundary_timestamp": authorization_result[
                    "authorization_boundary_timestamp"
                ],
                "attempt_index": expected_index,
                "attempt_id": source["attempt_id"],
                "market_id": source["market_id"],
                "market_family": MARKET_FAMILY,
                "market_horizon_seconds": MARKET_HORIZON_SECONDS,
                "market_start_ts": market_start_ts,
                "decision_ts": decision_ts,
                "available_at_ts": available_at_ts,
                "max_input_ts": max_input_ts,
                "feature_cutoff_ts": feature_cutoff_ts,
                "paired_executable_ask_capture_attempted": True,
                "missing_values_encoded_as_numeric_zero": False,
                "complement_quote_proxy_used": False,
                "raw_evidence_manifest_index_path": source["raw_evidence_manifest_index_path"],
                "raw_evidence_manifest_index_sha256": source["raw_evidence_manifest_index_sha256"],
                "raw_evidence_manifest_sha256": source["raw_evidence_manifest_sha256"],
                "quality_valid": quality["quality_valid"],
                "direct_failure_reason_codes": quality["direct_failure_reason_codes"],
                "derived_failure_reason_codes": quality["derived_failure_reason_codes"],
                "quality_failure_reason_codes": quality["quality_failure_reason_codes"],
            }
        )
    output_path = Path(normalized_output_path).resolve()
    _write_new_jsonl(output_path, normalized_rows)
    return {
        "normalized_attempt_ledger_path": output_path,
        "normalized_attempt_ledger_sha256": sha256_file(output_path),
        "attempt_count": len(normalized_rows),
        "maximum_attempt_index": max(int(row["attempt_index"]) for row in normalized_rows),
        "quality_valid_count": sum(int(row["quality_valid"]) for row in normalized_rows),
        "authorization": authorization_result,
        "normalized_rows": normalized_rows,
        "window_incomplete_failed": False,
        "outcome_accessed": False,
        "safety": dict(SAFETY),
    }


def select_exact_authorized_window(
    *,
    normalized_attempt_ledger_path: Path | str,
    expected_normalized_attempt_ledger_sha256: str,
    authorization: AuthorizationExpectation,
    repository_root: Path | str | None = None,
) -> dict[str, Any]:
    """Select exact first 800 valid markets or return deterministic exhaustion."""

    repo_root = Path(repository_root or REPO_ROOT).resolve()
    authorization_result = validate_manual_collection_authorization(
        authorization,
        repository_root=repo_root,
    )
    ledger_path = Path(normalized_attempt_ledger_path).resolve()
    if sha256_file(ledger_path) != expected_normalized_attempt_ledger_sha256:
        raise ValueError("normalized attempt ledger SHA-256 mismatch")
    rows = _load_jsonl(ledger_path)
    if len(rows) > ATTEMPT_CAP:
        raise ValueError("attempt 927 is forbidden")
    if not rows:
        raise ValueError("normalized attempt ledger is empty")
    _validate_normalized_quality_against_raw_evidence(
        rows,
        repository_root=repo_root,
    )
    boundary_ms = int(authorization_result["authorization_boundary_timestamp_ms"])
    seen_attempt_ids: set[str] = set()
    seen_quality_markets: set[str] = set()
    accepted_in_attempt_order = []
    stop_attempt_index: int | None = None
    previous_ordering_key: tuple[int, str, int] | None = None
    for expected_index, row in enumerate(rows, start=1):
        _validate_normalized_row(
            row,
            expected_index=expected_index,
            authorization_result=authorization_result,
            boundary_ms=boundary_ms,
        )
        attempt_id = str(row["attempt_id"])
        if attempt_id in seen_attempt_ids:
            raise ValueError("duplicate attempt ID")
        seen_attempt_ids.add(attempt_id)
        ordering_key = (
            int(row["market_start_ts"]),
            str(row["market_id"]),
            expected_index,
        )
        if previous_ordering_key is not None and ordering_key < previous_ordering_key:
            raise ValueError("attempt ledger is not in chronological market order")
        previous_ordering_key = ordering_key
        if row["quality_valid"] is True:
            market_id = str(row["market_id"])
            if market_id in seen_quality_markets:
                raise ValueError("duplicate quality-valid market")
            seen_quality_markets.add(market_id)
            if stop_attempt_index is None:
                accepted_in_attempt_order.append(dict(row))
                if len(accepted_in_attempt_order) == TARGET_MARKET_COUNT:
                    stop_attempt_index = expected_index
    attempts_consumed = len(rows)
    if stop_attempt_index is None:
        exhausted = attempts_consumed == ATTEMPT_CAP
        return {
            "window_complete": False,
            "window_incomplete_failed": exhausted,
            "failure_reason": (
                "attempt_cap_exhausted_before_exact_800"
                if exhausted
                else "more_authorized_attempts_required"
            ),
            "attempts_consumed": attempts_consumed,
            "remaining_authorized_attempts": ATTEMPT_CAP - attempts_consumed,
            "quality_valid_unique_market_count": len(accepted_in_attempt_order),
            "selected_rows": [],
            "audit_only_post_boundary_rows": [],
            "extension_allowed": False,
            "replacement_allowed": False,
            "reset_allowed": False,
            "new_attempt_budget_allowed": False,
            "safety": dict(SAFETY),
        }
    selected = sorted(
        accepted_in_attempt_order,
        key=lambda row: (
            int(row["market_start_ts"]),
            str(row["market_id"]),
            int(row["attempt_index"]),
        ),
    )
    audit_only = [dict(row) for row in rows if int(row["attempt_index"]) > stop_attempt_index]
    return {
        "window_complete": True,
        "window_incomplete_failed": False,
        "failure_reason": None,
        "attempts_consumed": attempts_consumed,
        "remaining_authorized_attempts": ATTEMPT_CAP - attempts_consumed,
        "quality_valid_unique_market_count": len(seen_quality_markets),
        "selected_rows": selected,
        "selected_market_ids": [row["market_id"] for row in selected],
        "selected_attempt_indices": [int(row["attempt_index"]) for row in selected],
        "window_stop_attempt_index": stop_attempt_index,
        "audit_only_post_boundary_rows": audit_only,
        "audit_only_post_boundary_attempt_indices": [
            int(row["attempt_index"]) for row in audit_only
        ],
        "extension_allowed": False,
        "replacement_allowed": False,
        "reset_allowed": False,
        "new_attempt_budget_allowed": False,
        "safety": dict(SAFETY),
    }


def validate_exact_window_selection(
    *,
    selection: Mapping[str, Any],
    supplied_market_ids: Sequence[str],
) -> None:
    """Reject skips, replacements, subsets, extensions, and reordered windows."""

    if selection.get("window_complete") is not True:
        raise ValueError("exact window is not complete")
    expected = list(selection["selected_market_ids"])
    if len(supplied_market_ids) != TARGET_MARKET_COUNT:
        raise ValueError("supplied window must contain exactly 800 markets")
    if list(supplied_market_ids) != expected:
        raise ValueError("skipped, replaced, or out-of-order valid market")


def build_confirmatory_capture_manifest(
    *,
    authorization: AuthorizationExpectation,
    normalized_attempt_ledger_path: Path | str,
    expected_normalized_attempt_ledger_sha256: str,
    candidate_decision_rows_path: Path | str,
    expected_candidate_decision_rows_sha256: str,
    baseline_decision_rows_path: Path | str,
    expected_baseline_decision_rows_sha256: str,
    raw_evidence_manifest_index_path: Path | str,
    expected_raw_evidence_manifest_index_sha256: str,
    collector_protocol_path: Path | str,
    expected_collector_protocol_sha256: str,
    output_dir: Path | str,
    repository_root: Path | str | None = None,
    validation_fixture_only: bool = False,
) -> dict[str, Any]:
    """Build one immutable exact-population capture manifest."""

    repo_root = Path(repository_root or REPO_ROOT).resolve()
    authorization_result = validate_manual_collection_authorization(
        authorization,
        repository_root=repo_root,
    )
    if validation_fixture_only != authorization_result["test_fixture_only"]:
        raise ValueError("capture manifest fixture role mismatch")
    collector_path = Path(collector_protocol_path).resolve()
    if sha256_file(collector_path) != expected_collector_protocol_sha256:
        raise ValueError("collector protocol SHA-256 mismatch")
    frozen_collector_sha = authorization_result["frozen_artifacts"]["collector_protocol"]["sha256"]
    if expected_collector_protocol_sha256 != frozen_collector_sha:
        raise ValueError("collector protocol does not match authorization")
    selection = select_exact_authorized_window(
        normalized_attempt_ledger_path=normalized_attempt_ledger_path,
        expected_normalized_attempt_ledger_sha256=(expected_normalized_attempt_ledger_sha256),
        authorization=authorization,
        repository_root=repo_root,
    )
    if selection["window_complete"] is not True:
        raise ValueError("cannot freeze an incomplete confirmatory window")
    candidate_path = Path(candidate_decision_rows_path).resolve()
    baseline_path = Path(baseline_decision_rows_path).resolve()
    if sha256_file(candidate_path) != expected_candidate_decision_rows_sha256:
        raise ValueError("candidate decision rows SHA-256 mismatch")
    if sha256_file(baseline_path) != expected_baseline_decision_rows_sha256:
        raise ValueError("baseline decision rows SHA-256 mismatch")
    candidate_rows = _load_jsonl(candidate_path)
    baseline_rows = _load_jsonl(baseline_path)
    ordered_market_ids = list(selection["selected_market_ids"])
    _validate_decision_population(
        rows=candidate_rows,
        expected_market_ids=ordered_market_ids,
        label="candidate",
    )
    _validate_decision_population(
        rows=baseline_rows,
        expected_market_ids=ordered_market_ids,
        label="baseline",
    )
    raw_index_path = Path(raw_evidence_manifest_index_path).resolve()
    if sha256_file(raw_index_path) != expected_raw_evidence_manifest_index_sha256:
        raise ValueError("raw evidence manifest index SHA-256 mismatch")
    raw_records = _load_jsonl(raw_index_path)
    _validate_raw_evidence_index_records(
        raw_records,
        repository_root=repo_root,
    )
    normalized_rows = _load_jsonl(Path(normalized_attempt_ledger_path))
    _validate_normalized_quality_against_raw_evidence(
        normalized_rows,
        repository_root=repo_root,
    )
    if len(raw_records) != len(normalized_rows):
        raise ValueError("raw evidence manifest count mismatch")
    record_hashes = [record["record_sha256"] for record in raw_records]
    attempts_consumed = len(normalized_rows)
    manifest_payload = {
        "schema_version": "bigan-btc-15m-confirmatory-capture-manifest-r2",
        "lineage_id": LINEAGE_ID,
        "authorization_artifact_id": authorization_result["authorization_artifact_id"],
        "authorization_artifact_sha256": authorization_result["authorization_artifact_sha256"],
        "collector_protocol_sha256": expected_collector_protocol_sha256,
        "exact_market_count": TARGET_MARKET_COUNT,
        "ordered_market_ids": ordered_market_ids,
        "ordered_market_ids_sha256": canonical_json_sha256(ordered_market_ids),
        "ordered_attempt_indices": list(selection["selected_attempt_indices"]),
        "ordered_attempt_indices_sha256": canonical_json_sha256(
            selection["selected_attempt_indices"]
        ),
        "first_market_start_ts": int(selection["selected_rows"][0]["market_start_ts"]),
        "last_market_start_ts": int(selection["selected_rows"][-1]["market_start_ts"]),
        "strictly_later_authorization_boundary": authorization_result[
            "authorization_boundary_timestamp"
        ],
        "attempts_consumed": attempts_consumed,
        "unused_attempt_capacity": ATTEMPT_CAP - attempts_consumed,
        "cap_exhausted": attempts_consumed == ATTEMPT_CAP,
        "all_decisions_frozen": True,
        "candidate_decision_row_count": len(candidate_rows),
        "candidate_decision_rows_sha256": (expected_candidate_decision_rows_sha256),
        "baseline_decision_row_count": len(baseline_rows),
        "baseline_decision_rows_sha256": (expected_baseline_decision_rows_sha256),
        "paired_decision_population_sha256": canonical_json_sha256(
            {
                "market_ids": ordered_market_ids,
                "candidate_decision_rows_sha256": (expected_candidate_decision_rows_sha256),
                "baseline_decision_rows_sha256": (expected_baseline_decision_rows_sha256),
            }
        ),
        "raw_evidence_manifest_count": len(raw_records),
        "raw_evidence_manifest_set_sha256": canonical_json_sha256(record_hashes),
        "raw_evidence_manifest_index_sha256": (expected_raw_evidence_manifest_index_sha256),
        "normalized_attempt_ledger_sha256": (expected_normalized_attempt_ledger_sha256),
        "no_outcome_access_confirmation": True,
        "no_settlement_confirmation": True,
        "window_stop_attempt_index": selection["window_stop_attempt_index"],
        "audit_only_post_boundary_attempt_indices": selection[
            "audit_only_post_boundary_attempt_indices"
        ],
        "validation_fixture_only": validation_fixture_only,
        "promotion_evidence_eligible": False,
        "state": dict(STATE_BLOCKED),
        "safety": dict(SAFETY),
    }
    validate_capture_manifest_structure(manifest_payload)
    output_path = Path(output_dir).resolve() / "confirmatory_capture_manifest.json"
    frozen = _write_new_frozen_json(output_path, manifest_payload)
    return {
        "capture_manifest_path": frozen["path"],
        "capture_manifest_sha256": frozen["sha256"],
        "payload": manifest_payload,
        "selection": selection,
    }


def validate_capture_manifest_structure(payload: Mapping[str, Any]) -> None:
    """Validate required capture-manifest fields without trusting gate booleans."""

    missing = sorted(CAPTURE_MANIFEST_REQUIRED_FIELDS - set(payload))
    if missing:
        raise ValueError("capture manifest missing fields: " + ",".join(missing))
    if payload["lineage_id"] != LINEAGE_ID:
        raise ValueError("capture manifest lineage mismatch")
    if int(payload["exact_market_count"]) != TARGET_MARKET_COUNT:
        raise ValueError("capture manifest must contain exactly 800 markets")
    market_ids = payload["ordered_market_ids"]
    attempt_indices = payload["ordered_attempt_indices"]
    if not isinstance(market_ids, list) or len(market_ids) != TARGET_MARKET_COUNT:
        raise ValueError("capture manifest ordered market count mismatch")
    if len(set(market_ids)) != TARGET_MARKET_COUNT:
        raise ValueError("capture manifest contains duplicate markets")
    if not isinstance(attempt_indices, list) or len(attempt_indices) != TARGET_MARKET_COUNT:
        raise ValueError("capture manifest ordered attempt count mismatch")
    if canonical_json_sha256(market_ids) != payload["ordered_market_ids_sha256"]:
        raise ValueError("capture manifest market ID hash mismatch")
    if canonical_json_sha256(attempt_indices) != payload["ordered_attempt_indices_sha256"]:
        raise ValueError("capture manifest attempt index hash mismatch")
    attempts_consumed = int(payload["attempts_consumed"])
    if not TARGET_MARKET_COUNT <= attempts_consumed <= ATTEMPT_CAP:
        raise ValueError("capture manifest attempts consumed is invalid")
    if int(payload["unused_attempt_capacity"]) != ATTEMPT_CAP - attempts_consumed:
        raise ValueError("capture manifest unused attempt capacity mismatch")
    if bool(payload["cap_exhausted"]) != (attempts_consumed == ATTEMPT_CAP):
        raise ValueError("capture manifest cap exhausted mismatch")
    if int(payload["candidate_decision_row_count"]) != TARGET_MARKET_COUNT:
        raise ValueError("candidate decision row count mismatch")
    if int(payload["baseline_decision_row_count"]) != TARGET_MARKET_COUNT:
        raise ValueError("baseline decision row count mismatch")
    for field in (
        "authorization_artifact_sha256",
        "collector_protocol_sha256",
        "candidate_decision_rows_sha256",
        "baseline_decision_rows_sha256",
        "paired_decision_population_sha256",
        "raw_evidence_manifest_set_sha256",
        "raw_evidence_manifest_index_sha256",
        "normalized_attempt_ledger_sha256",
    ):
        _require_sha256(str(payload[field]), field)
    if payload["no_outcome_access_confirmation"] is not True:
        raise ValueError("capture manifest outcome-access confirmation missing")
    if payload["no_settlement_confirmation"] is not True:
        raise ValueError("capture manifest settlement confirmation missing")
    if payload["promotion_evidence_eligible"] is not False:
        raise ValueError("capture manifest cannot be promotion evidence yet")
    if payload["safety"] != SAFETY or any(payload["safety"].values()):
        raise ValueError("capture manifest safety flags must remain false")


def validate_and_authorize_exact_outcome_access(
    *,
    authorization: AuthorizationExpectation,
    capture_manifest_path: Path | str,
    expected_capture_manifest_sha256: str,
    normalized_attempt_ledger_path: Path | str,
    expected_normalized_attempt_ledger_sha256: str,
    candidate_decision_rows_path: Path | str,
    expected_candidate_decision_rows_sha256: str,
    baseline_decision_rows_path: Path | str,
    expected_baseline_decision_rows_sha256: str,
    raw_evidence_manifest_index_path: Path | str,
    expected_raw_evidence_manifest_index_sha256: str,
    collector_protocol_path: Path | str,
    expected_collector_protocol_sha256: str,
    statistical_protocol_path: Path | str,
    expected_statistical_protocol_sha256: str,
    requested_market_ids: Sequence[str],
    repository_root: Path | str | None = None,
) -> dict[str, Any]:
    """Independently verify exact frozen artifacts before one full outcome open."""

    if requested_market_ids is None:
        raise ValueError("requested_market_ids is mandatory")
    repo_root = Path(repository_root or REPO_ROOT).resolve()
    authorization_result = validate_manual_collection_authorization(
        authorization,
        repository_root=repo_root,
    )
    manifest_path = Path(capture_manifest_path).resolve()
    _verify_sidecar(manifest_path)
    if sha256_file(manifest_path) != expected_capture_manifest_sha256:
        raise ValueError("capture manifest SHA-256 mismatch")
    manifest = _load_json_object(manifest_path)
    validate_capture_manifest_structure(manifest)
    if (
        manifest["authorization_artifact_sha256"]
        != authorization_result["authorization_artifact_sha256"]
    ):
        raise ValueError("capture manifest authorization mismatch")

    collector_path = Path(collector_protocol_path).resolve()
    statistical_path = Path(statistical_protocol_path).resolve()
    _verify_exact_hash(
        collector_path,
        expected_collector_protocol_sha256,
        "collector protocol",
    )
    _verify_exact_hash(
        statistical_path,
        expected_statistical_protocol_sha256,
        "statistical protocol",
    )
    if (
        expected_collector_protocol_sha256
        != authorization_result["frozen_artifacts"]["collector_protocol"]["sha256"]
    ):
        raise ValueError("collector protocol differs from authorization")
    if (
        expected_statistical_protocol_sha256
        != authorization_result["frozen_artifacts"]["statistical_protocol"]["sha256"]
    ):
        raise ValueError("statistical protocol differs from authorization")
    if manifest["collector_protocol_sha256"] != expected_collector_protocol_sha256:
        raise ValueError("capture manifest collector protocol mismatch")

    ledger_path = Path(normalized_attempt_ledger_path).resolve()
    _verify_exact_hash(
        ledger_path,
        expected_normalized_attempt_ledger_sha256,
        "normalized attempt ledger",
    )
    if manifest["normalized_attempt_ledger_sha256"] != expected_normalized_attempt_ledger_sha256:
        raise ValueError("capture manifest normalized ledger mismatch")
    normalized_rows = _load_jsonl(ledger_path)
    _validate_normalized_ledger_for_outcome(
        normalized_rows,
        authorization_result=authorization_result,
    )
    selection = select_exact_authorized_window(
        normalized_attempt_ledger_path=ledger_path,
        expected_normalized_attempt_ledger_sha256=(expected_normalized_attempt_ledger_sha256),
        authorization=authorization,
        repository_root=repo_root,
    )
    if selection["window_complete"] is not True:
        raise ValueError("outcome access requires a complete exact window")
    ordered_market_ids = list(selection["selected_market_ids"])
    if list(requested_market_ids) != ordered_market_ids:
        raise ValueError("partial, reordered, or incremental outcome opening")
    if manifest["ordered_market_ids"] != ordered_market_ids:
        raise ValueError("capture manifest market population mismatch")
    if manifest["ordered_attempt_indices"] != selection["selected_attempt_indices"]:
        raise ValueError("capture manifest attempt population mismatch")
    attempts_consumed = len(normalized_rows)
    if int(manifest["attempts_consumed"]) != attempts_consumed:
        raise ValueError("capture manifest attempts-consumed mismatch")
    if int(manifest["unused_attempt_capacity"]) != ATTEMPT_CAP - attempts_consumed:
        raise ValueError("capture manifest unused-capacity mismatch")
    boundary_ms = int(authorization_result["authorization_boundary_timestamp_ms"])
    if int(manifest["first_market_start_ts"]) <= boundary_ms:
        raise ValueError("capture population is not strictly later")
    if int(manifest["first_market_start_ts"]) != int(
        selection["selected_rows"][0]["market_start_ts"]
    ):
        raise ValueError("capture manifest first market timestamp mismatch")
    if int(manifest["last_market_start_ts"]) != int(
        selection["selected_rows"][-1]["market_start_ts"]
    ):
        raise ValueError("capture manifest last market timestamp mismatch")

    candidate_path = Path(candidate_decision_rows_path).resolve()
    baseline_path = Path(baseline_decision_rows_path).resolve()
    _verify_exact_hash(
        candidate_path,
        expected_candidate_decision_rows_sha256,
        "candidate decision rows",
    )
    _verify_exact_hash(
        baseline_path,
        expected_baseline_decision_rows_sha256,
        "baseline decision rows",
    )
    if manifest["candidate_decision_rows_sha256"] != expected_candidate_decision_rows_sha256:
        raise ValueError("candidate decision hash mismatch")
    if manifest["baseline_decision_rows_sha256"] != expected_baseline_decision_rows_sha256:
        raise ValueError("baseline decision hash mismatch")
    candidate_rows = _load_jsonl(candidate_path)
    baseline_rows = _load_jsonl(baseline_path)
    _validate_decision_population(
        rows=candidate_rows,
        expected_market_ids=ordered_market_ids,
        label="candidate",
    )
    _validate_decision_population(
        rows=baseline_rows,
        expected_market_ids=ordered_market_ids,
        label="baseline",
    )
    paired_sha = canonical_json_sha256(
        {
            "market_ids": ordered_market_ids,
            "candidate_decision_rows_sha256": (expected_candidate_decision_rows_sha256),
            "baseline_decision_rows_sha256": (expected_baseline_decision_rows_sha256),
        }
    )
    if manifest["paired_decision_population_sha256"] != paired_sha:
        raise ValueError("paired decision population hash mismatch")

    raw_index_path = Path(raw_evidence_manifest_index_path).resolve()
    _verify_exact_hash(
        raw_index_path,
        expected_raw_evidence_manifest_index_sha256,
        "raw evidence manifest index",
    )
    if (
        manifest["raw_evidence_manifest_index_sha256"]
        != expected_raw_evidence_manifest_index_sha256
    ):
        raise ValueError("raw evidence index hash mismatch")
    raw_records = _load_jsonl(raw_index_path)
    _validate_raw_evidence_index_records(
        raw_records,
        repository_root=repo_root,
    )
    _validate_normalized_quality_against_raw_evidence(
        normalized_rows,
        repository_root=repo_root,
    )
    if int(manifest["raw_evidence_manifest_count"]) != len(raw_records):
        raise ValueError("raw evidence manifest count mismatch")
    raw_set_sha = canonical_json_sha256([record["record_sha256"] for record in raw_records])
    if manifest["raw_evidence_manifest_set_sha256"] != raw_set_sha:
        raise ValueError("raw evidence manifest set hash mismatch")
    validation_fixture_only = authorization_result["test_fixture_only"]
    return {
        "outcome_access_validation_passed": True,
        "exact_full_population_authorized": not validation_fixture_only,
        "validation_fixture_only": validation_fixture_only,
        "gate_would_authorize_if_non_fixture": validation_fixture_only,
        "outcome_opened_by_validator": False,
        "requested_market_count": len(requested_market_ids),
        "candidate_market_count": len(candidate_rows),
        "baseline_market_count": len(baseline_rows),
        "duplicate_market_count": 0,
        "dropped_market_count": 0,
        "out_of_window_market_count": 0,
        "attempts_consumed": attempts_consumed,
        "authorization_artifact_sha256": authorization_result["authorization_artifact_sha256"],
        "capture_manifest_sha256": expected_capture_manifest_sha256,
        "normalized_attempt_ledger_sha256": (expected_normalized_attempt_ledger_sha256),
        "candidate_decision_rows_sha256": (expected_candidate_decision_rows_sha256),
        "baseline_decision_rows_sha256": (expected_baseline_decision_rows_sha256),
        "raw_evidence_manifest_index_sha256": (expected_raw_evidence_manifest_index_sha256),
        "safety": dict(SAFETY),
    }


def build_r2_protocol_artifacts(
    *,
    repository_root: Path | str | None = None,
    created_at: str = CREATED_AT,
) -> dict[str, Any]:
    """Freeze versioned runtime-boundary contracts without granting authority."""

    repo_root = Path(repository_root or REPO_ROOT).resolve()
    v2_dir = repo_root / "examples" / "v8" / "polymarket_configs" / LINEAGE_ID
    graph = _load_verified_json(v2_dir / "moe_artifact_graph.json")
    if graph["bundle_hash"] != CANDIDATE_BUNDLE_HASH:
        raise ValueError("candidate bundle hash changed")
    module_descriptor = _descriptor(Path(__file__), repository_root=repo_root)
    health = _write_health_reason_semantics_r2(
        v2_dir=v2_dir,
        created_at=created_at,
    )
    collector_payload = {
        "schema_version": "bigan-btc-15m-moe-collector-boundary-r2",
        "lineage_id": LINEAGE_ID,
        "revision_id": REVISION_ID,
        "created_at": created_at,
        "candidate_bundle_hash": CANDIDATE_BUNDLE_HASH,
        "target_quality_valid_market_count": TARGET_MARKET_COUNT,
        "maximum_attempts": ATTEMPT_CAP,
        "authorization_validation_required_before_capture": True,
        "authorization_scope": "outcome_blind_capture_only",
        "integrated_attempt_ledger_validator": {
            "implementation": ("validate_authorized_attempt_ledger"),
            "source": module_descriptor,
            "caller_quality_valid_trusted": False,
            "raw_evidence_required": True,
            "attempt_hash_chain_required": True,
        },
        "quality_valid_derivation": {
            "contract": ("all frozen raw quality observations must be exactly true"),
            "required_observation_fields": list(QUALITY_OBSERVATION_FIELDS),
            "caller_quality_valid_trusted": False,
            "selector_rederives_quality_from_raw_evidence": True,
            "outcome_gate_rederives_quality_from_raw_evidence": True,
        },
        "exact_window_selector": {
            "implementation": "select_exact_authorized_window",
            "ordering": [
                "market_start_ts ascending",
                "market_id ascending",
                "attempt_index ascending",
            ],
            "strictly_later_than_authorization_boundary": True,
            "exact_market_count": TARGET_MARKET_COUNT,
            "attempt_927_forbidden": True,
            "post_exact_800_concurrent_attempts_are_audit_only": True,
            "cap_exhaustion_is_terminal_incomplete_failure": True,
            "extension_replacement_skip_reset_or_new_budget_allowed": False,
        },
        "required_raw_streams": list(REQUIRED_RAW_STREAMS),
        "causality": {
            "available_at_ts_lte_decision_ts": True,
            "max_input_ts_lte_decision_ts": True,
            "feature_cutoff_ts_lte_decision_ts": True,
            "missing_numeric_zero_forbidden": True,
            "complement_quote_proxy_forbidden": True,
        },
        "supersedes_for_runtime_collection": _descriptor(
            v2_dir / "moe_confirmatory_collector_protocol_r1.json",
            repository_root=repo_root,
        ),
        "health_reason_semantics_amendment": _descriptor(
            health["path"],
            repository_root=repo_root,
        ),
        "state": {
            **STATE_BLOCKED,
            "collector_runtime_enabled": False,
            "authorization_artifact_present": False,
        },
        "safety": dict(SAFETY),
    }
    collector = _write_new_frozen_json(
        v2_dir / "moe_confirmatory_collector_protocol_r2.json",
        collector_payload,
    )
    outcome_payload = {
        "schema_version": "bigan-btc-15m-moe-outcome-access-boundary-r2",
        "lineage_id": LINEAGE_ID,
        "revision_id": REVISION_ID,
        "created_at": created_at,
        "implementation": {
            "function": "validate_and_authorize_exact_outcome_access",
            "source": module_descriptor,
        },
        "collector_protocol": _descriptor(
            collector["path"],
            repository_root=repo_root,
        ),
        "statistical_protocol": _descriptor(
            v2_dir / "moe_confirmatory_protocol_r1.json",
            repository_root=repo_root,
        ),
        "required_pinned_artifacts": [
            "manual_authorization_artifact",
            "confirmatory_capture_manifest_and_sidecar",
            "normalized_attempt_ledger",
            "candidate_decision_rows",
            "baseline_decision_rows",
            "raw_evidence_manifest_index",
            "collector_protocol",
            "statistical_protocol",
        ],
        "derived_not_trusted": [
            "all_artifact_hashes_reconcile",
            "decision_artifacts_frozen",
            "quality_valid",
        ],
        "exact_population_requirements": {
            "ordered_market_count": TARGET_MARKET_COUNT,
            "candidate_decision_row_count": TARGET_MARKET_COUNT,
            "baseline_decision_row_count": TARGET_MARKET_COUNT,
            "duplicate_market_count": 0,
            "dropped_market_count": 0,
            "out_of_window_market_count": 0,
            "requested_market_ids_required": True,
            "partial_reordered_subset_or_incremental_opening_forbidden": True,
        },
        "state": {
            **STATE_BLOCKED,
            "outcome_access_runtime_enabled": False,
        },
        "safety": dict(SAFETY),
    }
    outcome = _write_new_frozen_json(
        v2_dir / "moe_exact_outcome_access_protocol_r2.json",
        outcome_payload,
    )
    authorization_contract_payload = {
        "schema_version": "bigan-btc-15m-moe-manual-authorization-contract-r2",
        "lineage_id": LINEAGE_ID,
        "revision_id": REVISION_ID,
        "created_at": created_at,
        "implementation": {
            "function": "validate_manual_collection_authorization",
            "source": module_descriptor,
        },
        "required_frozen_artifact_keys": list(AUTHORIZATION_FROZEN_KEYS),
        "collector_protocol": _descriptor(
            collector["path"],
            repository_root=repo_root,
        ),
        "outcome_access_protocol": _descriptor(
            outcome["path"],
            repository_root=repo_root,
        ),
        "maximum_attempts": ATTEMPT_CAP,
        "exact_market_target": TARGET_MARKET_COUNT,
        "only_permitted_scope": "outcome_blind_capture_only",
        "forbidden_permissions": [
            "outcome_access",
            "settlement",
            "training",
            "paper",
            "promotion",
            "live_trading",
            "wallet_signing",
            "polymarket_writes",
        ],
        "contract_is_collection_authority": False,
        "manual_artifact_still_required": True,
        "state": dict(STATE_BLOCKED),
        "safety": dict(SAFETY),
    }
    authorization_contract = _write_new_frozen_json(
        v2_dir / "moe_manual_collection_authorization_contract_r2.json",
        authorization_contract_payload,
    )
    frozen_for_template = frozen_authorization_artifacts(repo_root)
    authorization_template_payload = {
        "schema_version": "bigan-btc-15m-moe-manual-authorization-r2",
        "lineage_id": LINEAGE_ID,
        "authorization_artifact_id": None,
        "authorization_request_text_sha256": None,
        "authorization_decision_text_sha256": None,
        "approver_identity": None,
        "authorization_source_id": None,
        "authorization_source_url": None,
        "authorization_timestamp": None,
        "strictly_later_than_timestamp": None,
        "maximum_attempts": ATTEMPT_CAP,
        "exact_market_target": TARGET_MARKET_COUNT,
        "candidate_bundle_hash": CANDIDATE_BUNDLE_HASH,
        "frozen_artifacts": frozen_for_template,
        "authorization_scope": "outcome_blind_capture_only",
        "explicit_manual_authorization": False,
        "outcome_blind_capture_authorized": False,
        "test_fixture_only": False,
        "permissions": {
            "outcome_access_enabled": False,
            "settlement_enabled": False,
            "training_enabled": False,
            "paper_enabled": False,
            "promotion_enabled": False,
            "live_trading_enabled": False,
            "wallet_signing_enabled": False,
            "polymarket_write_enabled": False,
        },
        "state": dict(STATE_BLOCKED),
        "safety": dict(SAFETY),
        "artifact_role": ("inactive_copy-only_template_not_collection_authority"),
        "template_usable_as_collection_authorization": False,
        "human_completion_requirements": [
            "copy this frozen template to a distinct auditable artifact",
            "fill every null identity, provenance, text-hash, and timestamp field",
            "set explicit_manual_authorization=true",
            "set outcome_blind_capture_authorized=true",
            "set state.fresh_collection_authorized=true",
            "recompute and externally pin the completed artifact SHA-256",
            "pass completed artifact and caller-pinned facts to the validator",
        ],
        "authorization_contract": _descriptor(
            authorization_contract["path"],
            repository_root=repo_root,
        ),
    }
    authorization_template = _write_new_frozen_json(
        v2_dir / "moe_manual_collection_authorization_template_r2.json",
        authorization_template_payload,
    )
    return {
        "health_reason_semantics": health,
        "collector_protocol": collector,
        "outcome_access_protocol": outcome,
        "authorization_contract": authorization_contract,
        "authorization_template": authorization_template,
        "candidate_bundle_hash": CANDIDATE_BUNDLE_HASH,
        "target_market_count": TARGET_MARKET_COUNT,
        "attempt_cap": ATTEMPT_CAP,
        "state": dict(STATE_BLOCKED),
        "safety": dict(SAFETY),
    }


def build_synthetic_boundary_validation_fixture(
    *,
    repository_root: Path | str | None = None,
    output_dir: Path | str,
    created_at: str = CREATED_AT,
) -> dict[str, Any]:
    """Build a non-authoritative synthetic 801-attempt end-to-end fixture."""

    repo_root = Path(repository_root or REPO_ROOT).resolve()
    fixture_root = Path(output_dir).resolve()
    if fixture_root.exists() and any(fixture_root.iterdir()):
        raise FileExistsError(f"fixture output already exists: {fixture_root}")
    fixture_root.mkdir(parents=True, exist_ok=True)
    stream_descriptors = {}
    for index, name in enumerate(REQUIRED_RAW_STREAMS, start=1):
        path = fixture_root / name
        _write_new_text_artifact(
            path,
            json.dumps(
                {
                    "schema_version": "synthetic-outcome-blind-stream-r2",
                    "stream_name": name,
                    "fixture_sequence": index,
                    "outcome_bearing": False,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
        )
        stream_descriptors[name] = _descriptor(
            path,
            repository_root=repo_root,
        )

    boundary = "2030-01-01T00:00:00+00:00"
    boundary_ms = _parse_timestamp_ms(boundary)
    raw_records = []
    for attempt_index in range(1, 802):
        market_id = f"synthetic-btc15m-{attempt_index:04d}"
        record = {
            "schema_version": "bigan-synthetic-raw-evidence-record-r2",
            "attempt_id": f"synthetic-attempt-{attempt_index:04d}",
            "market_id": market_id,
            "raw_streams": stream_descriptors,
            "paired_executable_ask_capture_attempted": True,
            "missing_values_encoded_as_numeric_zero": False,
            "complement_quote_proxy_used": False,
            "causality": {
                "available_at_ts": boundary_ms + attempt_index * 900_000 + 10,
                "decision_ts": boundary_ms + attempt_index * 900_000 + 300_000,
                "max_input_ts": boundary_ms + attempt_index * 900_000 + 20,
                "feature_cutoff_ts": boundary_ms + attempt_index * 900_000 + 30,
            },
            "quality_observations": {
                "market_identity_complete": True,
                "provider_capture_complete": True,
                "paired_executable_asks_complete": True,
                "book_capture_complete": True,
                "chainlink_capture_complete": True,
            },
            "direct_failure_reason_codes": [],
            "derived_failure_reason_codes": [],
        }
        record["record_sha256"] = canonical_json_sha256(record)
        raw_records.append(record)
    raw_index_path = fixture_root / "raw_evidence_manifest_index.jsonl"
    _write_new_jsonl(raw_index_path, raw_records)
    raw_index_descriptor = _descriptor(
        raw_index_path,
        repository_root=repo_root,
    )

    request_sha = "1" * 64
    decision_sha = "2" * 64
    authorization_payload = build_manual_authorization_payload(
        repository_root=repo_root,
        authorization_artifact_id="synthetic-boundary-validation-only",
        authorization_request_text_sha256=request_sha,
        authorization_decision_text_sha256=decision_sha,
        approver_identity="synthetic-test-approver",
        authorization_source_id="synthetic-test-source",
        authorization_source_url="https://example.invalid/synthetic-authorization",
        authorization_timestamp="2029-12-31T23:59:00+00:00",
        strictly_later_than_timestamp=boundary,
        test_fixture_only=True,
    )
    authorization_path = fixture_root / "manual_authorization_fixture.json"
    authorization_frozen = _write_new_frozen_json(
        authorization_path,
        authorization_payload,
    )
    authorization = AuthorizationExpectation(
        artifact_path=authorization_path,
        expected_artifact_sha256=authorization_frozen["sha256"],
        authorization_request_text_sha256=request_sha,
        authorization_decision_text_sha256=decision_sha,
        approver_identity="synthetic-test-approver",
        authorization_source_id="synthetic-test-source",
        authorization_source_url="https://example.invalid/synthetic-authorization",
        authorization_timestamp="2029-12-31T23:59:00+00:00",
        strictly_later_than_timestamp=boundary,
        expected_frozen_artifact_sha256={
            key: descriptor["sha256"]
            for key, descriptor in frozen_authorization_artifacts(repo_root).items()
        },
        allow_test_fixture=True,
    )
    attempt_rows = []
    previous = "0" * 64
    for attempt_index, evidence in enumerate(raw_records, start=1):
        market_start = boundary_ms + attempt_index * 900_000
        causality = evidence["causality"]
        row = {
            "attempt_index": attempt_index,
            "attempt_id": evidence["attempt_id"],
            "market_id": evidence["market_id"],
            "market_family": MARKET_FAMILY,
            "market_horizon_seconds": MARKET_HORIZON_SECONDS,
            "market_start_ts": market_start,
            "decision_ts": causality["decision_ts"],
            "available_at_ts": causality["available_at_ts"],
            "max_input_ts": causality["max_input_ts"],
            "feature_cutoff_ts": causality["feature_cutoff_ts"],
            "paired_executable_ask_capture_attempted": True,
            "missing_values_encoded_as_numeric_zero": False,
            "complement_quote_proxy_used": False,
            "raw_evidence_manifest_index_path": raw_index_descriptor["path"],
            "raw_evidence_manifest_index_sha256": raw_index_descriptor["sha256"],
            "raw_evidence_manifest_sha256": evidence["record_sha256"],
            "previous_entry_sha256": previous,
        }
        row["entry_sha256"] = canonical_json_sha256(row)
        previous = row["entry_sha256"]
        attempt_rows.append(row)
    source_ledger_path = fixture_root / "authorized_attempt_ledger.jsonl"
    _write_new_jsonl(source_ledger_path, attempt_rows)
    normalized_path = fixture_root / "normalized_attempt_ledger.jsonl"
    ledger_result = validate_authorized_attempt_ledger(
        authorization=authorization,
        attempt_ledger_path=source_ledger_path,
        expected_attempt_ledger_sha256=sha256_file(source_ledger_path),
        normalized_output_path=normalized_path,
        repository_root=repo_root,
    )
    selection = select_exact_authorized_window(
        normalized_attempt_ledger_path=normalized_path,
        expected_normalized_attempt_ledger_sha256=ledger_result["normalized_attempt_ledger_sha256"],
        authorization=authorization,
        repository_root=repo_root,
    )
    if selection["selected_attempt_indices"] != list(range(1, 801)):
        raise ValueError("synthetic exact window did not stop at attempt 800")
    if selection["audit_only_post_boundary_attempt_indices"] != [801]:
        raise ValueError("synthetic concurrent attempt 801 is not audit-only")

    candidate_rows = []
    baseline_rows = []
    for row in selection["selected_rows"]:
        common = {
            "market_id": row["market_id"],
            "attempt_index": row["attempt_index"],
            "decision_ts": row["decision_ts"],
            "outcome_accessed": False,
        }
        candidate_rows.append(
            {
                **common,
                "model_role": "candidate",
                "prediction": 0.51,
                "decision": "NO_TRADE",
            }
        )
        baseline_rows.append(
            {
                **common,
                "model_role": "matched_global_baseline",
                "prediction": 0.50,
                "decision": "NO_TRADE",
            }
        )
    candidate_path = fixture_root / "candidate_decision_rows.jsonl"
    baseline_path = fixture_root / "baseline_decision_rows.jsonl"
    _write_new_jsonl(candidate_path, candidate_rows)
    _write_new_jsonl(baseline_path, baseline_rows)
    v2_dir = repo_root / "examples" / "v8" / "polymarket_configs" / LINEAGE_ID
    collector_path = v2_dir / "moe_confirmatory_collector_protocol_r2.json"
    capture = build_confirmatory_capture_manifest(
        authorization=authorization,
        normalized_attempt_ledger_path=normalized_path,
        expected_normalized_attempt_ledger_sha256=sha256_file(normalized_path),
        candidate_decision_rows_path=candidate_path,
        expected_candidate_decision_rows_sha256=sha256_file(candidate_path),
        baseline_decision_rows_path=baseline_path,
        expected_baseline_decision_rows_sha256=sha256_file(baseline_path),
        raw_evidence_manifest_index_path=raw_index_path,
        expected_raw_evidence_manifest_index_sha256=sha256_file(raw_index_path),
        collector_protocol_path=collector_path,
        expected_collector_protocol_sha256=sha256_file(collector_path),
        output_dir=fixture_root,
        repository_root=repo_root,
        validation_fixture_only=True,
    )
    outcome = validate_and_authorize_exact_outcome_access(
        authorization=authorization,
        capture_manifest_path=capture["capture_manifest_path"],
        expected_capture_manifest_sha256=capture["capture_manifest_sha256"],
        normalized_attempt_ledger_path=normalized_path,
        expected_normalized_attempt_ledger_sha256=sha256_file(normalized_path),
        candidate_decision_rows_path=candidate_path,
        expected_candidate_decision_rows_sha256=sha256_file(candidate_path),
        baseline_decision_rows_path=baseline_path,
        expected_baseline_decision_rows_sha256=sha256_file(baseline_path),
        raw_evidence_manifest_index_path=raw_index_path,
        expected_raw_evidence_manifest_index_sha256=sha256_file(raw_index_path),
        collector_protocol_path=collector_path,
        expected_collector_protocol_sha256=sha256_file(collector_path),
        statistical_protocol_path=(v2_dir / "moe_confirmatory_protocol_r1.json"),
        expected_statistical_protocol_sha256=sha256_file(
            v2_dir / "moe_confirmatory_protocol_r1.json"
        ),
        requested_market_ids=selection["selected_market_ids"],
        repository_root=repo_root,
    )
    fixture_payload = {
        "schema_version": "bigan-btc-15m-boundary-validation-fixture-r2",
        "lineage_id": LINEAGE_ID,
        "created_at": created_at,
        "role": "synthetic_non_authoritative_runtime_validation_only",
        "is_manual_collection_authorization": False,
        "is_confirmatory_collection": False,
        "is_promotion_evidence": False,
        "candidate_bundle_hash": CANDIDATE_BUNDLE_HASH,
        "target_market_count": TARGET_MARKET_COUNT,
        "attempt_cap": ATTEMPT_CAP,
        "authorization_fixture": _descriptor(
            authorization_path,
            repository_root=repo_root,
        ),
        "source_attempt_ledger": _descriptor(
            source_ledger_path,
            repository_root=repo_root,
        ),
        "normalized_attempt_ledger": _descriptor(
            normalized_path,
            repository_root=repo_root,
        ),
        "raw_evidence_manifest_index": _descriptor(
            raw_index_path,
            repository_root=repo_root,
        ),
        "candidate_decision_rows": _descriptor(
            candidate_path,
            repository_root=repo_root,
        ),
        "baseline_decision_rows": _descriptor(
            baseline_path,
            repository_root=repo_root,
        ),
        "capture_manifest": _descriptor(
            capture["capture_manifest_path"],
            repository_root=repo_root,
        ),
        "authorization_validator_passed": True,
        "attempt_ledger_validator_passed": True,
        "exact_window_validator_passed": True,
        "capture_manifest_validator_passed": True,
        "outcome_access_gate_validation_passed": outcome["outcome_access_validation_passed"],
        "outcome_opened": False,
        "collection_started": False,
        "state": dict(STATE_BLOCKED),
        "safety": dict(SAFETY),
    }
    fixture_manifest = _write_new_frozen_json(
        fixture_root / "fixture_manifest.json",
        fixture_payload,
    )
    return {
        "fixture_manifest": fixture_manifest,
        "capture_manifest": capture,
        "ledger_result": ledger_result,
        "selection": selection,
        "outcome_validation": outcome,
        "candidate_decision_rows_sha256": sha256_file(candidate_path),
        "baseline_decision_rows_sha256": sha256_file(baseline_path),
        "normalized_attempt_ledger_sha256": sha256_file(normalized_path),
        "raw_evidence_manifest_index_sha256": sha256_file(raw_index_path),
        "state": dict(STATE_BLOCKED),
        "safety": dict(SAFETY),
    }


def build_final_boundary_attestation_r2(
    *,
    repository_root: Path | str,
    base_pytest_junit_path: Path | str,
    head_pytest_junit_path: Path | str,
    base_ruff_json_path: Path | str,
    head_ruff_json_path: Path | str,
    executable_head_commit: str,
    created_at: str = CREATED_AT,
) -> dict[str, Any]:
    """Write the Commit-B-only regression ledger and final boundary report."""

    repo_root = Path(repository_root).resolve()
    v2_dir = repo_root / "examples" / "v8" / "polymarket_configs" / LINEAGE_ID
    base_junit = Path(base_pytest_junit_path).resolve()
    head_junit = Path(head_pytest_junit_path).resolve()
    base_ruff_path = Path(base_ruff_json_path).resolve()
    head_ruff_path = Path(head_ruff_json_path).resolve()
    base_pytest = _parse_pytest_junit(base_junit)
    head_pytest = _parse_pytest_junit(head_junit)
    base_by_node = {row["node_id"]: row for row in base_pytest["failures"]}
    head_by_node = {row["node_id"]: row for row in head_pytest["failures"]}
    base_nodes = set(base_by_node)
    head_nodes = set(head_by_node)
    added = sorted(head_nodes - base_nodes)
    removed = sorted(base_nodes - head_nodes)
    changed = sorted(
        node
        for node in base_nodes & head_nodes
        if base_by_node[node]["normalized_message_sha256"]
        != head_by_node[node]["normalized_message_sha256"]
    )
    unchanged = sorted((base_nodes & head_nodes) - set(changed))
    base_ruff = _parse_ruff_json(base_ruff_path)
    head_ruff = _parse_ruff_json(head_ruff_path)
    base_ruff_ids = {row["identity"] for row in base_ruff}
    head_ruff_ids = {row["identity"] for row in head_ruff}
    added_ruff = sorted(head_ruff_ids - base_ruff_ids)
    ledger_payload = {
        "schema_version": "bigan-btc-15m-boundary-regression-ledger-r2",
        "lineage_id": LINEAGE_ID,
        "revision_id": REVISION_ID,
        "created_at": created_at,
        "base_commit": BASE_COMMIT,
        "executable_head_commit": executable_head_commit,
        "attestation_commit": None,
        "capture_hashes": {
            "base_pytest_junit_sha256": sha256_file(base_junit),
            "head_pytest_junit_sha256": sha256_file(head_junit),
            "base_ruff_json_sha256": sha256_file(base_ruff_path),
            "head_ruff_json_sha256": sha256_file(head_ruff_path),
        },
        "base_pytest": base_pytest,
        "head_pytest": head_pytest,
        "pytest_reconciliation": {
            "base_failure_node_ids": sorted(base_nodes),
            "head_failure_node_ids": sorted(head_nodes),
            "added_failure_node_ids": added,
            "removed_failure_node_ids": removed,
            "unchanged_failure_node_ids": unchanged,
            "changed_message_failure_node_ids": changed,
            "new_test_failure_count": len(added) + len(changed),
            "head_failures_subset_of_base_failures": not added and not changed,
        },
        "base_ruff_errors": base_ruff,
        "head_ruff_errors": head_ruff,
        "ruff_reconciliation": {
            "added_error_identities": added_ruff,
            "removed_error_identities": sorted(base_ruff_ids - head_ruff_ids),
            "unchanged_error_identities": sorted(base_ruff_ids & head_ruff_ids),
            "new_ruff_error_count": len(added_ruff),
        },
        "required_condition_passed": (not added and not changed and not added_ruff),
        "state": dict(STATE_BLOCKED),
        "safety": dict(SAFETY),
    }
    ledger = _write_new_frozen_json(
        v2_dir / "regression_failure_ledger_r2.json",
        ledger_payload,
    )
    fixture_dir = v2_dir / "collection_boundary_validation_fixture"
    fixture = _load_verified_json(fixture_dir / "fixture_manifest.json")
    final_payload = {
        "schema_version": "bigan-btc-15m-final-boundary-hardening-r2",
        "lineage_id": LINEAGE_ID,
        "revision_id": REVISION_ID,
        "created_at": created_at,
        "base_commit": BASE_COMMIT,
        "executable_head_commit": executable_head_commit,
        "attestation_commit": None,
        "candidate_bundle_hash": CANDIDATE_BUNDLE_HASH,
        "candidate_bundle_unchanged": True,
        "target_market_count": TARGET_MARKET_COUNT,
        "attempt_cap": ATTEMPT_CAP,
        "authorization_validator_result": True,
        "exact_window_validator_result": True,
        "cap_exhaustion_result": "terminal_incomplete_failure",
        "capture_manifest": fixture["capture_manifest"],
        "candidate_decision_rows": fixture["candidate_decision_rows"],
        "baseline_decision_rows": fixture["baseline_decision_rows"],
        "normalized_attempt_ledger": fixture["normalized_attempt_ledger"],
        "outcome_access_gate_result": True,
        "validation_fixture_only": True,
        "collection_did_not_start": True,
        "fresh_outcome_opened": False,
        "regression_failure_ledger": _descriptor(
            ledger["path"],
            repository_root=repo_root,
        ),
        "regression_required_condition_passed": ledger_payload["required_condition_passed"],
        "state": dict(STATE_BLOCKED),
        "safety": dict(SAFETY),
    }
    final_report = _write_new_frozen_json(
        v2_dir / "final_collection_boundary_hardening_report.json",
        final_payload,
    )
    return {
        "regression_ledger": ledger,
        "final_report": final_report,
        "required_condition_passed": ledger_payload["required_condition_passed"],
    }


def _write_health_reason_semantics_r2(
    *,
    v2_dir: Path,
    created_at: str,
) -> dict[str, Any]:
    source_path = v2_dir / "collection_attempt_health_snapshot.jsonl"
    rows = _load_jsonl(source_path)
    broad_derived_codes = {
        "book_causality_failed",
        "btc_candle_coverage_failed",
        "chainlink_capture_failed",
        "chainlink_causality_failed",
        "decision_row_count_not_2",
        "feature_rows_missing",
        "market_row_coverage_failed",
        "orderbook_full_window_coverage_failed",
        "paired_executable_ask_coverage_failed",
        "provider_orderbook_snapshot_coverage_failed",
    }
    records = []
    for row in rows:
        if row["quality_valid"] is True:
            continue
        direct = sorted(set(row["quality_failure_reason_codes"]) - broad_derived_codes)
        derived = []
        if row["provider_capture_complete"] is False:
            derived.append("provider_capture_incomplete")
        if row["causality_checks_passed"] is False:
            derived.append("causality_checks_failed")
        effective = sorted(set(direct + derived))
        unknown = not effective
        records.append(
            {
                "attempt_index": row["attempt_index"],
                "attempt_id": row["attempt_id"],
                "directly_observed_failure_reason_codes": direct,
                "derived_failure_reason_codes": derived,
                "unknown_failure_reason": unknown,
                "quality_failure_reason_codes": (effective if effective else ["unknown"]),
            }
        )
    if len(records) != 7:
        raise ValueError("health reason amendment must contain seven invalid attempts")
    if any(
        set(record["quality_failure_reason_codes"]) >= broad_derived_codes for record in records
    ):
        raise ValueError("broad common reasons remain attached to invalid attempts")
    payload = {
        "schema_version": "bigan-btc-15m-health-reason-semantics-r2",
        "lineage_id": LINEAGE_ID,
        "revision_id": REVISION_ID,
        "created_at": created_at,
        "role": "diagnostic_amendment_only",
        "source_snapshot": _descriptor(source_path),
        "source_attempted_market_count": 120,
        "source_quality_valid_market_count": 113,
        "attempt_cap_unchanged": ATTEMPT_CAP,
        "direct_reason_semantics": (
            "reason code directly present in immutable capture-time evidence"
        ),
        "derived_reason_semantics": (
            "minimal reason derived from a failed frozen quality observation"
        ),
        "unknown_reason_semantics": (
            "use quality_failure_reason_codes=['unknown'] when neither direct "
            "nor derived evidence identifies a reason"
        ),
        "invalid_attempt_records": records,
        "invalid_attempt_record_count": len(records),
        "old_broad_reason_attachment_is_authoritative": False,
        "state": dict(STATE_BLOCKED),
        "safety": dict(SAFETY),
    }
    return _write_new_frozen_json(
        v2_dir / "collection_attempt_health_reason_semantics_r2.json",
        payload,
    )


def _load_raw_evidence_record(
    source: Mapping[str, Any],
    *,
    repository_root: Path,
    index_cache: dict[tuple[str, str], dict[str, dict[str, Any]]],
    stream_hash_cache: dict[tuple[str, str], bool],
) -> dict[str, Any]:
    index_path_text = str(source["raw_evidence_manifest_index_path"])
    index_sha = str(source["raw_evidence_manifest_index_sha256"])
    _require_sha256(index_sha, "raw evidence manifest index")
    index_path = _resolve_repo_path(index_path_text, repository_root)
    cache_key = (index_path_text, index_sha)
    if cache_key not in index_cache:
        if sha256_file(index_path) != index_sha:
            raise ValueError("raw evidence manifest index SHA-256 mismatch")
        records = _load_jsonl(index_path)
        _validate_raw_evidence_index_records(
            records,
            repository_root=repository_root,
            stream_hash_cache=stream_hash_cache,
        )
        index_cache[cache_key] = {str(record["record_sha256"]): record for record in records}
    record_sha = str(source["raw_evidence_manifest_sha256"])
    _require_sha256(record_sha, "raw evidence manifest")
    try:
        return index_cache[cache_key][record_sha]
    except KeyError as error:
        raise ValueError("raw evidence manifest hash not found") from error


def _validate_raw_evidence_index_records(
    records: Sequence[Mapping[str, Any]],
    *,
    repository_root: Path,
    stream_hash_cache: dict[tuple[str, str], bool] | None = None,
) -> None:
    cache = stream_hash_cache if stream_hash_cache is not None else {}
    record_hashes = set()
    for record_raw in records:
        record = dict(record_raw)
        forbidden = _find_forbidden_keys(record, FORBIDDEN_ATTEMPT_FIELDS)
        if forbidden:
            raise ValueError(
                "raw evidence contains forbidden model/outcome field: " + ",".join(forbidden)
            )
        record_sha = str(record.pop("record_sha256", ""))
        _require_sha256(record_sha, "raw evidence record")
        if canonical_json_sha256(record) != record_sha:
            raise ValueError("raw evidence record SHA-256 mismatch")
        if record_sha in record_hashes:
            raise ValueError("duplicate raw evidence record hash")
        record_hashes.add(record_sha)
        streams = record.get("raw_streams")
        if not isinstance(streams, dict) or set(streams) != set(REQUIRED_RAW_STREAMS):
            raise ValueError("missing required raw stream")
        for name in REQUIRED_RAW_STREAMS:
            descriptor = streams[name]
            key = (str(descriptor.get("path")), str(descriptor.get("sha256")))
            if key not in cache:
                _verify_descriptor(descriptor, repository_root=repository_root)
                cache[key] = True
        causality = record.get("causality") or {}
        if not (
            int(causality["available_at_ts"]) <= int(causality["decision_ts"])
            and int(causality["max_input_ts"]) <= int(causality["decision_ts"])
            and int(causality["feature_cutoff_ts"]) <= int(causality["decision_ts"])
        ):
            raise ValueError("raw evidence causality violation")
        if record.get("paired_executable_ask_capture_attempted") is not True:
            raise ValueError("raw evidence did not attempt paired asks")
        if record.get("missing_values_encoded_as_numeric_zero") is not False:
            raise ValueError("raw evidence missing values encoded as zero")
        if record.get("complement_quote_proxy_used") is not False:
            raise ValueError("raw evidence complement quote proxy is forbidden")
        observations = record.get("quality_observations")
        if not isinstance(observations, dict):
            raise ValueError("raw evidence quality observations are required")
        if not isinstance(record.get("direct_failure_reason_codes"), list):
            raise ValueError("direct failure reason list is required")
        if not isinstance(record.get("derived_failure_reason_codes"), list):
            raise ValueError("derived failure reason list is required")
        _derive_quality_from_evidence(record)


def _derive_quality_from_evidence(
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive the frozen quality result only from verified raw observations."""

    observations = evidence.get("quality_observations")
    if not isinstance(observations, Mapping):
        raise ValueError("raw evidence quality observations are required")
    missing_observations = [
        field for field in QUALITY_OBSERVATION_FIELDS if field not in observations
    ]
    if missing_observations:
        raise ValueError(
            "raw evidence quality observations missing: " + ",".join(missing_observations)
        )
    for field in QUALITY_OBSERVATION_FIELDS:
        if type(observations[field]) is not bool:
            raise ValueError(f"raw quality observation is not boolean: {field}")
    quality_valid = all(observations[field] is True for field in QUALITY_OBSERVATION_FIELDS)
    direct_raw = evidence.get("direct_failure_reason_codes")
    derived_raw = evidence.get("derived_failure_reason_codes")
    if not isinstance(direct_raw, list) or not isinstance(derived_raw, list):
        raise ValueError("raw evidence failure reason lists are required")
    direct = sorted({str(value).strip() for value in direct_raw if str(value).strip()})
    derived = sorted({str(value).strip() for value in derived_raw if str(value).strip()})
    if quality_valid and (direct or derived):
        raise ValueError("quality-valid raw evidence contains failure reasons")
    return {
        "quality_valid": quality_valid,
        "direct_failure_reason_codes": direct,
        "derived_failure_reason_codes": derived,
        "quality_failure_reason_codes": (
            [] if quality_valid else (sorted(set(direct + derived)) or ["unknown"])
        ),
    }


def _validate_normalized_quality_against_raw_evidence(
    rows: Sequence[Mapping[str, Any]],
    *,
    repository_root: Path,
) -> None:
    """Reject a normalized quality result that raw evidence cannot reproduce."""

    index_cache: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    stream_hash_cache: dict[tuple[str, str], bool] = {}
    for row in rows:
        evidence = _load_raw_evidence_record(
            row,
            repository_root=repository_root,
            index_cache=index_cache,
            stream_hash_cache=stream_hash_cache,
        )
        if evidence.get("attempt_id") != row.get("attempt_id"):
            raise ValueError("normalized attempt ID does not match raw evidence")
        if evidence.get("market_id") != row.get("market_id"):
            raise ValueError("normalized market ID does not match raw evidence")
        causality = evidence.get("causality") or {}
        for field in (
            "available_at_ts",
            "decision_ts",
            "max_input_ts",
            "feature_cutoff_ts",
        ):
            if int(causality[field]) != int(row[field]):
                raise ValueError(f"normalized {field} does not match raw evidence")
        if evidence.get("paired_executable_ask_capture_attempted") is not True:
            raise ValueError("raw evidence did not attempt paired asks")
        if evidence.get("missing_values_encoded_as_numeric_zero") is not False:
            raise ValueError("raw evidence encodes missing as numeric zero")
        if evidence.get("complement_quote_proxy_used") is not False:
            raise ValueError("raw evidence uses complement quote proxy")
        quality = _derive_quality_from_evidence(evidence)
        for field in (
            "quality_valid",
            "direct_failure_reason_codes",
            "derived_failure_reason_codes",
            "quality_failure_reason_codes",
        ):
            if row.get(field) != quality[field]:
                raise ValueError(f"normalized {field} is not derived from raw evidence")


def _validate_source_attempt_hash_chain(
    rows: Sequence[Mapping[str, Any]],
) -> None:
    previous = "0" * 64
    for expected_index, raw in enumerate(rows, start=1):
        row = dict(raw)
        if int(row.get("attempt_index") or 0) != expected_index:
            raise ValueError("attempt indices must be contiguous from 1")
        if row.get("previous_entry_sha256") != previous:
            raise ValueError("attempt hash-chain predecessor mismatch")
        entry_sha = str(row.pop("entry_sha256", ""))
        _require_sha256(entry_sha, "attempt entry")
        if canonical_json_sha256(row) != entry_sha:
            raise ValueError("attempt hash-chain entry mismatch")
        previous = entry_sha


def _validate_normalized_row(
    row: Mapping[str, Any],
    *,
    expected_index: int,
    authorization_result: Mapping[str, Any],
    boundary_ms: int,
) -> None:
    forbidden = _find_forbidden_keys(row, FORBIDDEN_NORMALIZED_FIELDS)
    if forbidden:
        raise ValueError(
            "normalized ledger contains model or outcome field: " + ",".join(forbidden)
        )
    if row.get("schema_version") != "bigan-btc-15m-authorized-attempt-r2":
        raise ValueError("normalized attempt schema mismatch")
    if row.get("lineage_id") != LINEAGE_ID:
        raise ValueError("normalized attempt lineage mismatch")
    if int(row.get("attempt_index") or 0) != expected_index:
        raise ValueError("normalized attempt indices are not contiguous")
    if expected_index > ATTEMPT_CAP:
        raise ValueError("attempt 927 is forbidden")
    if (
        row.get("authorization_artifact_sha256")
        != authorization_result["authorization_artifact_sha256"]
    ):
        raise ValueError("normalized attempt authorization mismatch")
    if row.get("authorization_artifact_id") != authorization_result["authorization_artifact_id"]:
        raise ValueError("normalized attempt authorization ID mismatch")
    if (
        row.get("authorization_boundary_timestamp")
        != authorization_result["authorization_boundary_timestamp"]
    ):
        raise ValueError("normalized authorization boundary mismatch")
    if not str(row.get("attempt_id") or ""):
        raise ValueError("normalized attempt ID is required")
    if not str(row.get("market_id") or ""):
        raise ValueError("normalized market ID is required")
    if row.get("market_family") != MARKET_FAMILY:
        raise ValueError("normalized market family mismatch")
    if int(row.get("market_horizon_seconds") or 0) != MARKET_HORIZON_SECONDS:
        raise ValueError("normalized market horizon mismatch")
    if int(row.get("market_start_ts") or 0) <= boundary_ms:
        raise ValueError("normalized market is not strictly later")
    if row.get("paired_executable_ask_capture_attempted") is not True:
        raise ValueError("normalized paired ask attempt missing")
    if row.get("missing_values_encoded_as_numeric_zero") is not False:
        raise ValueError("normalized missing semantics mismatch")
    if row.get("complement_quote_proxy_used") is not False:
        raise ValueError("normalized complement proxy is forbidden")
    if type(row.get("quality_valid")) is not bool:
        raise ValueError("normalized quality result must be boolean")
    for field in (
        "direct_failure_reason_codes",
        "derived_failure_reason_codes",
        "quality_failure_reason_codes",
    ):
        if not isinstance(row.get(field), list):
            raise ValueError(f"normalized {field} must be a list")
    _require_sha256(
        str(row.get("raw_evidence_manifest_index_sha256") or ""),
        "normalized raw evidence index",
    )
    _require_sha256(
        str(row.get("raw_evidence_manifest_sha256") or ""),
        "normalized raw evidence record",
    )
    if not (
        int(row["available_at_ts"]) <= int(row["decision_ts"])
        and int(row["max_input_ts"]) <= int(row["decision_ts"])
        and int(row["feature_cutoff_ts"]) <= int(row["decision_ts"])
    ):
        raise ValueError("normalized attempt causality violation")


def _validate_normalized_ledger_for_outcome(
    rows: Sequence[Mapping[str, Any]],
    *,
    authorization_result: Mapping[str, Any],
) -> None:
    boundary_ms = int(authorization_result["authorization_boundary_timestamp_ms"])
    attempt_ids = set()
    for expected_index, row in enumerate(rows, start=1):
        _validate_normalized_row(
            row,
            expected_index=expected_index,
            authorization_result=authorization_result,
            boundary_ms=boundary_ms,
        )
        attempt_id = str(row["attempt_id"])
        if attempt_id in attempt_ids:
            raise ValueError("duplicate attempt ID")
        attempt_ids.add(attempt_id)
    if len(rows) > ATTEMPT_CAP:
        raise ValueError("attempt 927 is forbidden")


def _validate_decision_population(
    *,
    rows: Sequence[Mapping[str, Any]],
    expected_market_ids: Sequence[str],
    label: str,
) -> None:
    if len(rows) != TARGET_MARKET_COUNT:
        raise ValueError(f"{label} decision row count mismatch")
    forbidden = [
        path
        for row in rows
        for path in _find_forbidden_keys(
            row,
            FORBIDDEN_DECISION_OUTCOME_FIELDS,
        )
    ]
    if forbidden:
        raise ValueError(f"{label} decision rows contain outcome fields")
    market_ids = [str(row.get("market_id") or "") for row in rows]
    if len(set(market_ids)) != TARGET_MARKET_COUNT:
        raise ValueError(f"{label} decision rows contain duplicate markets")
    if market_ids != list(expected_market_ids):
        raise ValueError(f"{label} decision market population mismatch")


def _find_forbidden_keys(
    payload: Any,
    forbidden: set[str],
    *,
    prefix: str = "",
) -> list[str]:
    found = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            if key_text in forbidden:
                found.append(path)
            found.extend(_find_forbidden_keys(value, forbidden, prefix=path))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            found.extend(
                _find_forbidden_keys(
                    value,
                    forbidden,
                    prefix=f"{prefix}[{index}]",
                )
            )
    return sorted(found)


def _load_verified_json(path: Path) -> dict[str, Any]:
    _verify_sidecar(path)
    return _load_json_object(path)


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _load_jsonl(path: Path | str) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _verify_sidecar(path: Path) -> None:
    sidecar = path.with_suffix(".sha256")
    if not sidecar.exists():
        raise ValueError(f"missing SHA-256 sidecar: {path}")
    if sidecar.read_text(encoding="utf-8").strip() != sha256_file(path):
        raise ValueError(f"SHA-256 sidecar mismatch: {path}")


def _verify_exact_hash(path: Path, expected_sha256: str, label: str) -> None:
    _require_sha256(expected_sha256, label)
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch")


def _descriptor(
    path: Path | str,
    *,
    repository_root: Path | str | None = None,
) -> dict[str, str]:
    repo_root = Path(repository_root or REPO_ROOT).resolve()
    resolved = Path(path).resolve()
    return {
        "path": resolved.relative_to(repo_root).as_posix(),
        "sha256": sha256_file(resolved),
    }


def _verify_descriptor(
    descriptor: Mapping[str, Any],
    *,
    repository_root: Path,
) -> Path:
    if set(descriptor) != {"path", "sha256"}:
        raise ValueError("artifact descriptor field set mismatch")
    path = _resolve_repo_path(str(descriptor["path"]), repository_root)
    _verify_exact_hash(path, str(descriptor["sha256"]), "artifact descriptor")
    return path


def _resolve_repo_path(path_text: str, repository_root: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        raise ValueError("machine-local absolute path is forbidden")
    resolved = (repository_root / path).resolve()
    try:
        resolved.relative_to(repository_root)
    except ValueError as error:
        raise ValueError("artifact path escapes repository root") from error
    return resolved


def _write_new_frozen_json(
    path: Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if path.exists() or path.with_suffix(".sha256").exists():
        raise FileExistsError(f"refusing to overwrite frozen artifact: {path}")
    _atomic_write_text(
        path,
        json.dumps(
            dict(payload),
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
    )
    digest = sha256_file(path)
    _atomic_write_text(path.with_suffix(".sha256"), digest + "\n")
    return {"path": path, "sha256": digest}


def _write_new_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    if path.exists() or path.with_suffix(".sha256").exists():
        raise FileExistsError(f"refusing to overwrite frozen artifact: {path}")
    content = "".join(
        json.dumps(
            dict(row),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
        for row in rows
    )
    _atomic_write_text(path, content)
    _atomic_write_text(path.with_suffix(".sha256"), sha256_file(path) + "\n")


def _write_new_text_artifact(path: Path, content: str) -> None:
    if path.exists() or path.with_suffix(".sha256").exists():
        raise FileExistsError(f"refusing to overwrite frozen artifact: {path}")
    _atomic_write_text(path, content)
    _atomic_write_text(path.with_suffix(".sha256"), sha256_file(path) + "\n")


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _require_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256")


def _parse_timestamp_ms(value: str) -> int:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"invalid authorization timestamp: {value}") from error
    if parsed.tzinfo is None:
        raise ValueError("authorization timestamp must include a timezone")
    return int(parsed.astimezone(UTC).timestamp() * 1000)


def main() -> None:
    """Freeze the r2 contracts and their explicitly non-authorizing fixture."""

    repo_root = REPO_ROOT.resolve()
    v2_dir = repo_root / "examples" / "v8" / "polymarket_configs" / LINEAGE_ID
    protocols = build_r2_protocol_artifacts(repository_root=repo_root)
    fixture = build_synthetic_boundary_validation_fixture(
        repository_root=repo_root,
        output_dir=v2_dir / "collection_boundary_validation_fixture",
    )
    print(
        json.dumps(
            {
                "protocols": protocols,
                "fixture": fixture,
            },
            sort_keys=True,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
