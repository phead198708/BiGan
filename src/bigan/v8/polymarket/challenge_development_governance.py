"""Health, transfer, and training-readiness governance for the BTC-15m lane."""

from __future__ import annotations

import base64
import json
import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

from bigan.v8.polymarket.challenge_development_lane import (
    SAFETY,
    atomic_write_json,
    load_jsonl,
    sha256_file,
)
from bigan.v8.polymarket.contracts import canonical_json_sha256

HEALTH_SCHEMA_VERSION = "bigan-challenge-model-15m-daily-health-v1"
TRANSFER_PROTOCOL_SCHEMA_VERSION = "bigan-challenge-model-15m-transfer-diagnostic-protocol-v1"
TRAINING_PROTOCOL_SCHEMA_VERSION = "bigan-challenge-model-15m-training-preregistration-v1"
TRANSFER_REPORT_SCHEMA_VERSION = "bigan-challenge-model-15m-transfer-diagnostic-report-v1"
TRANSFER_FREEZE_SCHEMA_VERSION = "bigan-challenge-model-15m-transfer-diagnostic-freeze-v1"
READINESS_SCHEMA_VERSION = "bigan-challenge-model-15m-training-readiness-v1"
EXPECTED_DECISIONS_PER_MARKET = 2
REPO_ROOT = Path(__file__).resolve().parents[4]
V7_FEATURE_NAMES = (
    "spread",
    "market_implied_prob",
    "mid_price",
    "microprice",
    "obi_l1",
    "obi_l5",
    "obi_l10",
    "signed_volume_1m",
    "trade_imbalance_1m",
    "trade_count_1m",
    "trade_volume_1m",
    "ret_1m",
    "ret_5m",
    "ret_15m",
    "rv_1m",
    "rv_5m",
    "rv_15m",
    "minute_of_day",
    "day_of_week",
    "underlying_id",
    "horizon_minutes",
    "liquidity_bucket",
    "ret_30m",
    "rv_30m",
    "aggressor_buy_ratio_1m",
    "avg_trade_size_1m",
    "tick_spread",
    "tick_obi_l1",
    "tick_obi_l3",
    "tick_mid_price",
    "tick_price_velocity",
    "tick_trade_arrival_rate",
)
V8_1_FEATURE_NAMES = (
    "action_score_available",
    "action_score",
    "action_score_margin",
    "btc_anchor_direction",
    "selected_side_probability",
    "execution_price",
    "selected_side_probability_minus_execution_price",
    "log1p_spread_bps",
    "queue_fill_shortfall",
    "log1p_book_staleness_ms",
    "late_window_pressure",
    "pre_entry_market_exposure",
    "same_side_prior_entry",
    "side_flip_prior_entry",
    "side_is_up",
)
V8_1_NATIVE_FEATURE_NAMES = (
    "execution_price",
    "log1p_spread_bps",
    "queue_fill_shortfall",
    "log1p_book_staleness_ms",
    "late_window_pressure",
    "side_is_up",
)
V8_1_FORCED_MISSING_FEATURE_NAMES = (
    "action_score",
    "action_score_margin",
)
V8_1_BRIDGED_FEATURE_NAMES = (
    "action_score_available",
    "btc_anchor_direction",
    "selected_side_probability",
    "selected_side_probability_minus_execution_price",
    "pre_entry_market_exposure",
    "same_side_prior_entry",
    "side_flip_prior_entry",
)


def validate_transfer_protocol(
    protocol: Mapping[str, Any],
    *,
    verify_artifact_bytes: bool = True,
) -> None:
    """Reject semantic drift in the one-shot development transfer diagnostic."""

    trigger = dict(protocol.get("trigger") or {})
    quality = dict(protocol.get("quality_contract") or {})
    v7 = dict(protocol.get("legacy_v7_selected") or {})
    v81 = dict(protocol.get("v8_1_unit_controller") or {})
    reporting = dict(protocol.get("reporting") or {})
    weak_rule = dict(reporting.get("v8_1_weak_signal_rule") or {})
    artifact_descriptors = [
        dict(v7.get("model_artifact") or {}),
        dict(v7.get("settlement_model") or {}),
        dict(v7.get("settlement_residual_model") or {}),
        dict(v81.get("model_artifact") or {}),
    ]
    checks = {
        "schema": protocol.get("schema_version") == TRANSFER_PROTOCOL_SCHEMA_VERSION,
        "lane": protocol.get("lane_id") == "challenge-model-development-btc-updown-15m-v1",
        "role": protocol.get("development_only_forever") is True
        and protocol.get("promotion_evidence_eligible") is False,
        "trigger": trigger
        == {
            "minimum_quality_valid_outcome_finalized_market_count": 40,
            "run_count": 1,
            "market_selection": (
                "chronological_earliest_40_quality_valid_outcome_finalized_markets"
            ),
            "same_market_set_required_for_both_policies": True,
        },
        "quality": quality.get("expected_decision_rows_per_market") == 2
        and quality.get("paired_up_down_executable_ask_required_at_every_decision") is True
        and quality.get("book_and_chainlink_causality_required") is True
        and quality.get("missing_trade_tape_must_not_be_encoded_as_zero") is True,
        "v7_fixed": v7.get("policy_name") == "v7_selected_by_pnl_stability"
        and v7.get("minimum_confidence") == 0.75
        and v7.get("minimum_expected_edge") == 0.04
        and v7.get("true_paired_executable_asks_required") is True
        and v7.get("complement_quote_proxy_allowed") is False
        and v7.get("threshold_or_feature_search_allowed") is False,
        "v81_fixed": v81.get("candidate_name") == "adaptive_support_controller_v8_1"
        and v81.get("position_size") == 1.0
        and v81.get("source_action_score_available") is False
        and v81.get("source_action_score_missing_representation")
        == "xgboost_missing_nan_with_availability_flag_false"
        and v81.get("transfer_status") == "bridge_handicapped_not_native"
        and v81.get("native_feature_coverage_must_be_reported") is True
        and v81.get("late_window_pressure_denominator") == "market_horizon_seconds"
        and v81.get("expected_market_horizon_seconds") == 900
        and v81.get("threshold_or_feature_search_allowed") is False,
        "reporting": reporting.get("old_15m_plus_12_39_allowed_as_gate") is False
        and reporting.get("cross_policy_superiority_claim_allowed") is False
        and reporting.get("hold_to_settlement_and_sell_before_close_must_use_separate_panels")
        is True,
        "weak_rule": weak_rule
        == {
            "minimum_accepted_market_count": 5,
            "metric": "mean_unit_net_pnl_per_accepted_market",
            "bootstrap_resamples": 10000,
            "bootstrap_seed": 0,
            "one_sided_confidence": 0.95,
            "pass_condition": "accepted_market_count_gte_5_and_bootstrap_lcb_gt_0",
            "promotion_claim_allowed": False,
            "full_retrain_conclusion_allowed": False,
        },
        "repo_pinned_artifacts": all(
            _is_repo_content_addressed_descriptor(descriptor)
            for descriptor in artifact_descriptors
        ),
        "safety": protocol.get("safety") == SAFETY,
    }
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ValueError("15m transfer protocol invalid: " + ", ".join(blockers))
    if verify_artifact_bytes:
        for key in (
            "model_artifact",
            "settlement_model",
            "settlement_residual_model",
        ):
            descriptor = dict(v7.get(key) or {})
            _verify_descriptor(descriptor, f"legacy v7 {key}")
        _verify_descriptor(
            dict(v81.get("model_artifact") or {}),
            "v8.1 model artifact",
        )


def validate_training_protocol(protocol: Mapping[str, Any]) -> None:
    """Validate the locked model-layer training preregistration."""

    gate = dict(protocol.get("readiness_gate") or {})
    cap = dict(gate.get("collector_cap_consistency") or {})
    split = dict(protocol.get("split") or {})
    representation = dict(protocol.get("representation") or {})
    target = dict(protocol.get("target") or {})
    discipline = dict(protocol.get("development_discipline") or {})
    checks = {
        "schema": protocol.get("schema_version") == TRAINING_PROTOCOL_SCHEMA_VERSION,
        "lane": protocol.get("lane_id") == "challenge-model-development-btc-updown-15m-v1",
        "locked": protocol.get("training_start_locked") is True,
        "development_only": protocol.get("development_only_forever") is True
        and protocol.get("promotion_evidence_eligible") is False,
        "gate": gate
        == {
            "minimum_quality_valid_outcome_finalized_market_count": 100,
            "minimum_paired_executable_quote_coverage": 0.95,
            "transfer_diagnostic_must_be_complete_and_sha256_pinned": True,
            "all_conditions_required": True,
            "manual_or_implicit_override_allowed": False,
            "collector_cap_consistency": {
                "maximum_capture_attempts_without_additional_permission": 119,
                "minimum_required_must_not_exceed_collector_cap": True,
                "additional_collection_authorization_created": False,
            },
        },
        "reachable_under_cap": int(
            gate.get("minimum_quality_valid_outcome_finalized_market_count") or 0
        )
        <= int(cap.get("maximum_capture_attempts_without_additional_permission") or -1),
        "split": split.get("method") == "chronological_unique_market_groups"
        and split.get("all_rows_for_one_market_must_remain_in_one_split") is True
        and split.get("random_row_split_allowed") is False
        and split.get("future_to_past_leakage_allowed") is False,
        "symmetric": representation.get("shared_side_symmetric_model") is True
        and representation.get("side_specific_model_or_quota_allowed") is False,
        "missing": representation.get("missing_encoded_as_numeric_zero_allowed") is False,
        "target": target.get("target_available_only_post_market_close") is True
        and target.get("official_finalized_outcome_required") is True
        and target.get("target_allowed_in_feature_or_split_selection") is False
        and target.get("target_allowed_in_capture_control") is False,
        "prohibitions": discipline.get("old_15m_plus_12_39_allowed_as_gate") is False
        and discipline.get("five_minute_exact_195_threshold_mining_allowed") is False
        and discipline.get("development_lane_allowed_as_promotion_evidence") is False,
        "safety": protocol.get("safety") == SAFETY,
    }
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ValueError("15m training protocol invalid: " + ", ".join(blockers))


def validate_training_collector_cap_consistency(
    training_protocol: Mapping[str, Any],
    lane_protocol: Mapping[str, Any],
) -> dict[str, int]:
    """Prove the frozen training minimum is reachable under the lane authorization."""

    validate_training_protocol(training_protocol)
    gate = dict(training_protocol["readiness_gate"])
    registered = dict(gate["collector_cap_consistency"])
    authorization = dict(lane_protocol.get("authorization_checkpoint") or {})
    minimum = int(gate["minimum_quality_valid_outcome_finalized_market_count"])
    registered_cap = int(registered["maximum_capture_attempts_without_additional_permission"])
    lane_cap = int(
        authorization.get("maximum_capture_attempts_before_additional_permission") or -1
    )
    checks = {
        "same_cap": registered_cap == lane_cap,
        "reachable": minimum <= lane_cap,
        "lane_stops_before_attempt_120": authorization.get("stop_before_attempt_120") is True,
        "lane_has_no_120_authorization": (
            authorization.get("explicit_120_round_authorization_recorded") is False
        ),
        "training_has_no_additional_authorization": (
            registered.get("additional_collection_authorization_created") is False
        ),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ValueError("training readiness conflicts with collector cap: " + ", ".join(blockers))
    return {
        "minimum_quality_valid_outcome_finalized_market_count": minimum,
        "maximum_capture_attempts_without_additional_permission": lane_cap,
    }


def audit_capture(capture: Mapping[str, Any]) -> dict[str, Any]:
    """Audit one outcome-blind capture without opening its target."""

    run_dir = Path(str(capture.get("run_dir") or "")).resolve()
    feature_path = run_dir / "phase2_corpus" / "polymarket_feature_rows.jsonl"
    capture_report_path = run_dir / "pending_round_capture_report.json"
    rows = load_jsonl(feature_path)
    reasons: list[str] = []
    if capture.get("capture_start_boundary_validation_passed") is not True:
        reasons.append("capture_start_boundary_failed")
    if int(capture.get("scheduled_round_start_ts") or 0) <= 0:
        reasons.append("capture_chronology_missing")
    if int(capture.get("raw_polymarket_market_count") or 0) != 1:
        reasons.append("market_row_coverage_failed")
    if int(capture.get("provider_raw_orderbook_snapshot_count") or 0) <= 0:
        reasons.append("provider_orderbook_snapshot_coverage_failed")
    if capture.get("orderbook_full_window_coverage_passed") is not True:
        reasons.append("orderbook_full_window_coverage_failed")
    if int(capture.get("raw_btc_candle_row_count") or 0) <= 0:
        reasons.append("btc_candle_coverage_failed")
    if (
        int(capture.get("raw_chainlink_price_row_count") or 0) <= 0
        or capture.get("chainlink_rtds_price_stream_fresh") is not True
    ):
        reasons.append("chainlink_capture_failed")
    if len(rows) != EXPECTED_DECISIONS_PER_MARKET:
        reasons.append("decision_row_count_not_2")
    if not feature_path.is_file():
        reasons.append("feature_rows_missing")

    paired = 0
    book_causal = 0
    chainlink_causal = 0
    trade_tape_causal = 0
    for row in rows:
        decision_ts = int(row.get("decision_ts") or 0)
        features = dict(row.get("features") or {})
        provenance = dict(row.get("feature_provenance") or {})
        globally_causal = (
            decision_ts > 0
            and int(row.get("available_at_ts") or 0) <= decision_ts
            and int(row.get("max_input_ts") or 0) <= decision_ts
        )
        pair_passed = all(
            _is_executable_price(features.get(name)) for name in ("up_ask", "down_ask")
        )
        paired += int(pair_passed)
        book_passed = (
            globally_causal
            and pair_passed
            and all(
                _provenance_causal(provenance.get(name), decision_ts)
                for name in ("up_ask", "down_ask")
            )
        )
        book_causal += int(book_passed)
        chainlink_passed = globally_causal and all(
            _chainlink_provenance_causal(provenance.get(name), decision_ts)
            for name in (
                "chainlink_price_at_decision",
                "chainlink_reference_price_at_market_start",
            )
        )
        chainlink_causal += int(chainlink_passed)
        tape_passed = (
            globally_causal
            and features.get("recent_trade_volume_coverage_complete") in (True, 1)
            and int(features.get("trade_tape_available_at_ts") or 0) <= decision_ts
            and int(features.get("trade_tape_max_causal_input_ts") or 0) <= decision_ts
            and not bool(features.get("trade_tape_provider_timeout"))
            and not bool(features.get("trade_tape_truncated"))
            and not bool(features.get("trade_tape_censored"))
            and not bool(features.get("trade_tape_historical_backfill"))
        )
        trade_tape_causal += int(tape_passed)
    if paired != EXPECTED_DECISIONS_PER_MARKET:
        reasons.append("paired_executable_ask_coverage_failed")
    if book_causal != EXPECTED_DECISIONS_PER_MARKET:
        reasons.append("book_causality_failed")
    if chainlink_causal != EXPECTED_DECISIONS_PER_MARKET:
        reasons.append("chainlink_causality_failed")
    reasons.extend(_capture_safety_reasons(capture_report_path))
    for reason, count in dict(capture.get("reject_reason_counts") or {}).items():
        if int(count or 0) > 0:
            reasons.append(f"capture_reject_{reason}")
    unique_reasons = sorted(set(reasons))
    return {
        "run_id": str(capture.get("run_id") or ""),
        "run_dir": str(run_dir),
        "scheduled_round_start_ts": int(capture.get("scheduled_round_start_ts") or 0),
        "quality_valid": not unique_reasons,
        "exclusion_reason_codes": unique_reasons,
        "expected_decision_count": EXPECTED_DECISIONS_PER_MARKET,
        "observed_decision_count": len(rows),
        "paired_executable_ask_decision_count": paired,
        "book_causal_complete_decision_count": book_causal,
        "chainlink_causal_complete_decision_count": chainlink_causal,
        "trade_tape_causal_complete_decision_count": trade_tape_causal,
        "trade_tape_incomplete_is_explicit_missing_not_zero": True,
        "target_or_outcome_opened": False,
        "development_only_forever": True,
        "promotion_evidence_eligible": False,
        "safety": dict(SAFETY),
    }


def build_lane_health_summary(
    *,
    lane_root: Path | str,
    date_utc: str | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Build one cumulative and UTC-day health report from target-free captures."""

    root = Path(lane_root).resolve()
    selected_date = date_utc or datetime.now(UTC).date().isoformat()
    captures: list[dict[str, Any]] = []
    for batch in load_jsonl(root / "outcome_blind_capture_batch_index.jsonl"):
        summary_path = Path(str(batch["batch_summary_path"])).resolve()
        summary = _load_json(summary_path)
        for raw_capture in list(summary.get("captures") or []):
            capture = dict(raw_capture)
            capture["_collected_at"] = str(batch.get("collected_at") or "")
            captures.append(capture)
    audits = [audit_capture(capture) for capture in captures]
    finalized_ids = {
        str(row["run_id"]) for row in load_jsonl(root / "finalized_development_corpus_index.jsonl")
    }
    day_indices = [
        index
        for index, capture in enumerate(captures)
        if str(capture.get("_collected_at") or "").startswith(selected_date)
    ]
    collector_state = _optional_json(root / "collector_state.json")
    finalizer_state = _optional_json(root / "finalizer_state.json")
    report = {
        "schema_version": HEALTH_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "date_utc": selected_date,
        "lane_root": str(root),
        "daily": _health_rollup(
            [audits[index] for index in day_indices],
            finalized_ids=finalized_ids,
        ),
        "cumulative": _health_rollup(audits, finalized_ids=finalized_ids),
        "authorization": {
            "maximum_attempts_before_additional_permission": 119,
            "attempt_120_authorized": False,
            "pause_before_attempt_120_required": True,
            "collector_reported_attempted_market_count": int(
                collector_state.get("attempted_market_count") or 0
            ),
            "collector_status": collector_state.get("status"),
        },
        "service_health": {
            "collector_pid": collector_state.get("collector_pid"),
            "finalizer_pid": finalizer_state.get("finalizer_pid"),
            "collector_state_updated_at": collector_state.get("updated_at"),
            "finalizer_state_updated_at": finalizer_state.get("updated_at"),
            "finalizer_error_count": int(finalizer_state.get("error_count") or 0),
        },
        "outcomes_labels_or_pnl_read_for_health": False,
        "development_only_forever": True,
        "promotion_evidence_eligible": False,
        "safety": dict(SAFETY),
    }
    if int(report["cumulative"]["attempted_market_count"]) > 119:
        raise ValueError("development lane exceeded the 119-attempt authorization")
    if write:
        daily_path = root / "daily_health_summaries" / f"{selected_date}.json"
        atomic_write_json(daily_path, report)
        atomic_write_json(root / "development_lane_health_latest.json", report)
        _write_text(
            daily_path.with_suffix(".md"),
            _health_markdown(report),
        )
    return report


def run_transfer_diagnostic_if_ready(
    *,
    lane_root: Path | str,
    protocol_path: Path | str,
    expected_protocol_sha256: str,
) -> dict[str, Any]:
    """Run the frozen same-window transfer diagnostic exactly once at 40 markets."""

    root = Path(lane_root).resolve()
    path = Path(protocol_path).resolve()
    if sha256_file(path) != expected_protocol_sha256.lower():
        raise ValueError("transfer protocol SHA-256 mismatch")
    protocol = _load_json(path)
    validate_transfer_protocol(protocol, verify_artifact_bytes=False)
    output_root = root / "transfer_diagnostic"
    freeze_path = output_root / "transfer_diagnostic_freeze.json"
    if freeze_path.is_file():
        freeze = _load_json(freeze_path)
        _validate_transfer_freeze(freeze)
        return {
            "status": "transfer_diagnostic_already_complete",
            "freeze_path": str(freeze_path),
            "freeze_sha256": sha256_file(freeze_path),
            "market_count": int(freeze["market_count"]),
        }
    captures = _indexed_capture_audits(root)
    finalized = {
        str(row["run_id"]): row
        for row in load_jsonl(root / "finalized_development_corpus_index.jsonl")
    }
    eligible = [
        audit for audit in captures if audit["quality_valid"] and audit["run_id"] in finalized
    ]
    eligible.sort(key=lambda row: (row["scheduled_round_start_ts"], row["run_id"]))
    minimum = int(
        (protocol.get("trigger") or {})["minimum_quality_valid_outcome_finalized_market_count"]
    )
    if len(eligible) < minimum:
        return {
            "status": "waiting_for_40_quality_valid_outcome_finalized_markets",
            "eligible_market_count": len(eligible),
            "required_market_count": minimum,
            "transfer_diagnostic_started": False,
            "outcomes_opened_by_transfer_diagnostic": False,
        }
    validate_transfer_protocol(protocol, verify_artifact_bytes=True)
    selected = eligible[:minimum]
    bundles = [_load_finalized_bundle(audit, finalized[audit["run_id"]]) for audit in selected]
    v7_rows, v7_summary = _run_v7_selected_transfer(
        bundles,
        dict(protocol["legacy_v7_selected"]),
    )
    v81_rows, v81_summary = _run_v8_1_transfer(
        bundles,
        dict(protocol["v8_1_unit_controller"]),
    )
    weak_signal = _weak_signal_assessment(
        v81_summary,
        dict(protocol["reporting"]["v8_1_weak_signal_rule"]),
    )
    market_ids = [str(bundle["market_id"]) for bundle in bundles]
    report = {
        "schema_version": TRANSFER_REPORT_SCHEMA_VERSION,
        "diagnostic_id": protocol["diagnostic_id"],
        "created_at": datetime.now(UTC).isoformat(),
        "protocol_path": str(path),
        "protocol_sha256": expected_protocol_sha256.lower(),
        "market_selection": protocol["trigger"]["market_selection"],
        "market_count": len(bundles),
        "market_ids": market_ids,
        "market_ids_sha256": canonical_json_sha256(market_ids),
        "same_market_set_used_for_both_policies": True,
        "policy_panels": {
            "legacy_v7_hold_to_settlement": {
                "lifecycle_policy": "HOLD_TO_SETTLEMENT",
                "metrics": v7_summary,
                "report_only": True,
                "binary_positive_pnl_gate_used": False,
            },
            "v8_1_sell_before_close_bridge": {
                "lifecycle_policy": "SELL_BEFORE_CLOSE",
                "transfer_status": "bridge_handicapped_not_native",
                "native_feature_coverage": v81_summary["native_feature_coverage"],
                "metrics": v81_summary,
                "weak_signal_rule": weak_signal,
                "full_retrain_conclusion_allowed": False,
                "promotion_claim_allowed": False,
            },
        },
        "cross_policy_comparison": {
            "superiority_claim_allowed": False,
            "superiority_claim_made": False,
            "reason": (
                "HOLD_TO_SETTLEMENT and SELL_BEFORE_CLOSE have different lifecycle "
                "targets and are reported in separate panels"
            ),
        },
        "old_15m_plus_12_39_used_as_gate": False,
        "threshold_or_feature_search_performed": False,
        "outcomes_opened_only_from_official_post_close_finalized_development_corpora": True,
        "development_only_forever": True,
        "promotion_evidence_eligible": False,
        "safety": dict(SAFETY),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    v7_path = output_root / "legacy_v7_hold_to_settlement_market_rows.jsonl"
    v81_path = output_root / "v8_1_sell_before_close_bridge_market_rows.jsonl"
    report_path = output_root / "transfer_diagnostic_report.json"
    _write_jsonl(v7_path, v7_rows)
    _write_jsonl(v81_path, v81_rows)
    atomic_write_json(report_path, report)
    _write_text(report_path.with_suffix(".md"), _transfer_markdown(report))
    freeze = {
        "schema_version": TRANSFER_FREEZE_SCHEMA_VERSION,
        "frozen_at": datetime.now(UTC).isoformat(),
        "protocol": _descriptor(path),
        "report": _descriptor(report_path),
        "legacy_v7_hold_to_settlement_market_rows": _descriptor(v7_path),
        "v8_1_sell_before_close_bridge_market_rows": _descriptor(v81_path),
        "market_count": len(bundles),
        "market_ids_sha256": report["market_ids_sha256"],
        "run_count": 1,
        "development_only_forever": True,
        "promotion_evidence_eligible": False,
        "safety": dict(SAFETY),
    }
    freeze["freeze_id"] = canonical_json_sha256(freeze)
    atomic_write_json(freeze_path, freeze)
    return {
        "status": "transfer_diagnostic_complete",
        "freeze_path": str(freeze_path),
        "freeze_sha256": sha256_file(freeze_path),
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
        "market_count": len(bundles),
    }


def build_training_readiness(
    *,
    lane_root: Path | str,
    training_protocol_path: Path | str,
    expected_training_protocol_sha256: str,
    transfer_protocol_path: Path | str,
    expected_transfer_protocol_sha256: str,
    write: bool = True,
) -> dict[str, Any]:
    """Evaluate but never bypass the preregistered model-training start gate."""

    root = Path(lane_root).resolve()
    training_path = Path(training_protocol_path).resolve()
    transfer_path = Path(transfer_protocol_path).resolve()
    if sha256_file(training_path) != expected_training_protocol_sha256.lower():
        raise ValueError("training protocol SHA-256 mismatch")
    if sha256_file(transfer_path) != expected_transfer_protocol_sha256.lower():
        raise ValueError("transfer protocol SHA-256 mismatch")
    training = _load_json(training_path)
    transfer = _load_json(transfer_path)
    validate_training_protocol(training)
    validate_transfer_protocol(transfer, verify_artifact_bytes=False)
    gate = dict(training["readiness_gate"])
    required_count = int(gate["minimum_quality_valid_outcome_finalized_market_count"])
    required_coverage = float(gate["minimum_paired_executable_quote_coverage"])
    collector_cap = int(
        gate["collector_cap_consistency"][
            "maximum_capture_attempts_without_additional_permission"
        ]
    )
    audits = _indexed_capture_audits(root)
    finalized_ids = {
        str(row["run_id"]) for row in load_jsonl(root / "finalized_development_corpus_index.jsonl")
    }
    finalized_audits = [row for row in audits if row["run_id"] in finalized_ids]
    quality_finalized = [row for row in finalized_audits if row["quality_valid"]]
    paired_numerator = sum(
        int(row["paired_executable_ask_decision_count"]) for row in finalized_audits
    )
    paired_denominator = len(finalized_audits) * EXPECTED_DECISIONS_PER_MARKET
    paired_coverage = paired_numerator / paired_denominator if paired_denominator else 0.0
    freeze_path = root / "transfer_diagnostic" / "transfer_diagnostic_freeze.json"
    transfer_complete = False
    transfer_freeze_sha256 = None
    if freeze_path.is_file():
        freeze = _load_json(freeze_path)
        _validate_transfer_freeze(freeze)
        transfer_complete = True
        transfer_freeze_sha256 = sha256_file(freeze_path)
    quality_condition = (
        f"quality_valid_outcome_finalized_market_count_at_least_{required_count}"
    )
    conditions = {
        quality_condition: len(quality_finalized) >= required_count,
        "paired_executable_quote_coverage_at_least_95_percent": (
            paired_coverage >= required_coverage
        ),
        "training_protocol_sha256_pinned_and_valid": True,
        "transfer_diagnostic_complete_and_sha256_pinned": transfer_complete,
    }
    blockers = [name for name, passed in conditions.items() if not passed]
    report = {
        "schema_version": READINESS_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "training_protocol": _descriptor(training_path),
        "transfer_protocol": _descriptor(transfer_path),
        "transfer_diagnostic_freeze_path": str(freeze_path),
        "transfer_diagnostic_freeze_sha256": transfer_freeze_sha256,
        "metrics": {
            "attempted_market_count": len(audits),
            "outcome_finalized_market_count": len(finalized_audits),
            "quality_valid_outcome_finalized_market_count": len(quality_finalized),
            "required_quality_valid_outcome_finalized_market_count": required_count,
            "maximum_authorized_capture_attempts": collector_cap,
            "readiness_count_reachable_under_collector_cap": required_count <= collector_cap,
            "paired_executable_quote_decision_count": paired_numerator,
            "paired_executable_quote_expected_decision_count": paired_denominator,
            "paired_executable_quote_coverage": paired_coverage,
            "required_paired_executable_quote_coverage": required_coverage,
        },
        "conditions": conditions,
        "blocking_reason_codes": blockers,
        "training_start_allowed": not blockers,
        "attempt_120_authorized": False,
        "authorization_note": (
            f"The readiness minimum ({required_count}) is reachable under the collector's "
            f"{collector_cap}-attempt cap. Attempt 120 remains unauthorized."
        ),
        "model_training_started": False,
        "old_15m_plus_12_39_used_as_gate": False,
        "five_minute_exact_195_threshold_mining_performed": False,
        "development_only_forever": True,
        "promotion_evidence_eligible": False,
        "safety": dict(SAFETY),
    }
    if write:
        atomic_write_json(root / "training_readiness_latest.json", report)
        _write_text(
            root / "training_readiness_latest.md",
            _readiness_markdown(report),
        )
    return report


def _indexed_capture_audits(root: Path) -> list[dict[str, Any]]:
    audits: list[dict[str, Any]] = []
    for batch in load_jsonl(root / "outcome_blind_capture_batch_index.jsonl"):
        summary = _load_json(Path(str(batch["batch_summary_path"])).resolve())
        audits.extend(audit_capture(row) for row in list(summary.get("captures") or []))
    return audits


def _health_rollup(
    audits: Sequence[Mapping[str, Any]],
    *,
    finalized_ids: set[str],
) -> dict[str, Any]:
    exclusions: Counter[str] = Counter()
    primary: Counter[str] = Counter()
    for row in audits:
        reasons = list(row.get("exclusion_reason_codes") or [])
        exclusions.update(reasons)
        if reasons:
            primary[reasons[0]] += 1
    expected = len(audits) * EXPECTED_DECISIONS_PER_MARKET
    quality = [row for row in audits if row.get("quality_valid") is True]
    return {
        "attempted_market_count": len(audits),
        "quality_valid_market_count": len(quality),
        "quality_valid_rate": len(quality) / len(audits) if audits else 0.0,
        "quality_valid_outcome_finalized_market_count": sum(
            str(row.get("run_id") or "") in finalized_ids for row in quality
        ),
        "outcome_finalized_market_count": sum(
            str(row.get("run_id") or "") in finalized_ids for row in audits
        ),
        "exclusion_reason_distribution_multi_label": dict(sorted(exclusions.items())),
        "exclusion_primary_reason_distribution": dict(sorted(primary.items())),
        "paired_up_down_executable_ask": _coverage(
            sum(int(row.get("paired_executable_ask_decision_count") or 0) for row in audits),
            expected,
        ),
        "causal_completeness": {
            "book": _coverage(
                sum(int(row.get("book_causal_complete_decision_count") or 0) for row in audits),
                expected,
            ),
            "chainlink": _coverage(
                sum(
                    int(row.get("chainlink_causal_complete_decision_count") or 0) for row in audits
                ),
                expected,
            ),
            "trade_tape": _coverage(
                sum(
                    int(row.get("trade_tape_causal_complete_decision_count") or 0) for row in audits
                ),
                expected,
            ),
        },
    }


def _coverage(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "complete_decision_count": numerator,
        "expected_decision_count": denominator,
        "coverage": numerator / denominator if denominator else 0.0,
    }


def _capture_safety_reasons(path: Path) -> list[str]:
    if not path.is_file():
        return ["capture_report_missing"]
    report = _load_json(path)
    expected = {
        "resolution_provider_called": False,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "live_exchange_write_enabled": False,
        "broker_exchange_write_enabled": False,
    }
    return [f"capture_safety_{key}" for key, value in expected.items() if report.get(key) != value]


def _provenance_causal(raw: Any, decision_ts: int) -> bool:
    value = dict(raw or {})
    return (
        int(value.get("available_at_ts") or 0) <= decision_ts
        and int(value.get("input_end_ts") or 0) <= decision_ts
    )


def _chainlink_provenance_causal(raw: Any, decision_ts: int) -> bool:
    value = dict(raw or {})
    return (
        value.get("provenance_valid") is True
        and int(value.get("available_at_ts") or 0) <= decision_ts
        and int(value.get("max_input_ts") or 0) <= decision_ts
    )


def _is_executable_price(value: Any) -> bool:
    try:
        price = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(price) and 0.0 < price < 1.0


def _load_finalized_bundle(
    audit: Mapping[str, Any],
    finalized: Mapping[str, Any],
) -> dict[str, Any]:
    corpus_manifest_path = Path(str(finalized["exported_corpus_manifest_path"])).resolve()
    corpus_dir = corpus_manifest_path.parent
    features = load_jsonl(corpus_dir / "polymarket_feature_rows.jsonl")
    labels = load_jsonl(corpus_dir / "polymarket_label_rows.jsonl")
    if len(features) != EXPECTED_DECISIONS_PER_MARKET:
        raise ValueError(f"finalized feature row count invalid: {audit['run_id']}")
    market_ids = {str(row.get("market_id") or "") for row in features}
    if len(market_ids) != 1:
        raise ValueError(f"finalized market identity invalid: {audit['run_id']}")
    return {
        "run_id": str(audit["run_id"]),
        "market_id": market_ids.pop(),
        "run_dir": Path(str(audit["run_dir"])).resolve(),
        "corpus_dir": corpus_dir,
        "features": sorted(features, key=lambda row: int(row["decision_ts"])),
        "labels": labels,
    }


class _LegacyV7:
    def __init__(self, protocol: Mapping[str, Any]) -> None:
        model_path = _resolve_pinned_path(protocol["model_artifact"]["path"])
        artifact = _load_json(model_path)
        if artifact.get("schema_version") != "xgboost_v7_settlement_ev_v1":
            raise ValueError("legacy v7 artifact schema invalid")
        if tuple(artifact.get("feature_columns") or ()) != V7_FEATURE_NAMES:
            raise ValueError("legacy v7 feature identity invalid")
        settlement_path = _resolve_pinned_path(protocol["settlement_model"]["path"])
        self.booster = xgb.Booster()
        self.booster.load_model(str(settlement_path))
        self.temperature = float(
            (artifact.get("calibration") or {})
            .get("family_temperatures", {})
            .get("BTC-15M", (artifact.get("calibration") or {})["global_temperature"])
        )

    def predict(self, values: Mapping[str, Any]) -> tuple[float, float, float]:
        vector = np.asarray(
            [[_nan_if_missing(values.get(name)) for name in V7_FEATURE_NAMES]],
            dtype=float,
        )
        raw = np.asarray(
            self.booster.predict(
                xgb.DMatrix(
                    vector,
                    missing=np.nan,
                    feature_names=list(V7_FEATURE_NAMES),
                )
            ),
            dtype=float,
        ).reshape(-1)
        if len(raw) != 3:
            raise ValueError("legacy v7 settlement output shape invalid")
        probabilities = np.maximum(raw, 1e-15)
        scaled = np.exp(np.log(probabilities) / self.temperature)
        scaled /= np.sum(scaled)
        return tuple(float(value) for value in scaled)


def _run_v7_selected_transfer(
    bundles: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model = _LegacyV7(protocol)
    selected_rows: list[dict[str, Any]] = []
    for bundle in bundles:
        chosen: dict[str, Any] | None = None
        for feature_row in bundle["features"]:
            candidates = []
            for side in ("UP", "DOWN"):
                bridge, coverage = _v7_feature_bridge(
                    feature_row,
                    side=side,
                    run_dir=Path(bundle["run_dir"]),
                )
                p_up, p_down, p_neutral = model.predict(bridge)
                features = dict(feature_row["features"])
                ask = float(features[f"{side.lower()}_ask"])
                mid = float(features[f"{side.lower()}_mid"])
                worst = min(0.99, ask + 0.02)
                p_side = p_up if side == "UP" else p_down
                expected_edge = p_side - worst
                if p_side >= 0.75 and expected_edge >= 0.04:
                    candidates.append(
                        {
                            "side": side,
                            "p_side": p_side,
                            "p_up": p_up,
                            "p_down": p_down,
                            "p_neutral": p_neutral,
                            "entry_ask": ask,
                            "entry_mid": mid,
                            "selection_worst_price": worst,
                            "expected_edge": expected_edge,
                            "feature_bridge_finite_count": coverage,
                            "feature_bridge_total_count": len(V7_FEATURE_NAMES),
                        }
                    )
            if candidates:
                chosen = max(
                    candidates,
                    key=lambda row: (
                        row["expected_edge"],
                        row["p_side"],
                        -row["entry_ask"],
                        row["side"] == "UP",
                    ),
                )
                chosen["decision_ts"] = int(feature_row["decision_ts"])
                break
        if chosen is None:
            selected_rows.append(
                _no_trade_row(bundle, policy="legacy_v7_hold_to_settlement")
            )
            continue
        label = _label(
            bundle,
            decision_ts=int(chosen["decision_ts"]),
            action=f"BUY_{chosen['side']}_HOLD_TO_SETTLEMENT",
        )
        selected_rows.append(
            _cost_row(
                bundle,
                policy="legacy_v7_hold_to_settlement",
                decision=chosen,
                label=label,
                predicted_gross_signal=float(chosen["p_side"]) - float(chosen["entry_mid"]),
            )
        )
    return selected_rows, _policy_summary(selected_rows)


def _run_v8_1_transfer(
    bundles: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    artifact = _load_json(_resolve_pinned_path(protocol["model_artifact"]["path"]))
    weighted = dict(artifact.get("final_weighted_model") or {})
    if tuple(weighted.get("feature_names") or ()) != V8_1_FEATURE_NAMES:
        raise ValueError("v8.1 scorer feature identity invalid")
    if str(weighted.get("booster_sha256") or "") != str(protocol["booster_sha256"]):
        raise ValueError("v8.1 scorer booster identity invalid")
    encoded = str(weighted["booster_json_base64"])
    if canonical_json_sha256(encoded) != str(protocol["booster_sha256"]):
        raise ValueError("v8.1 embedded booster SHA-256 mismatch")
    booster = xgb.Booster()
    booster.load_model(bytearray(base64.b64decode(encoded)))
    state = dict(artifact.get("final_rank_state") or {})
    scores = [float(value) for value in state["eligible_prediction_scores"]]
    guard_history = [bool(value) for value in state["controller_guard_acceptance_history"]]
    selected_rows: list[dict[str, Any]] = []
    for bundle in bundles:
        feature_row = bundle["features"][0]
        quantile, band = _controller_quantile(guard_history)
        threshold = _finite_sample_quantile(scores[-60:], quantile)
        side_scores = {}
        for side in ("UP", "DOWN"):
            bridge = _v8_1_feature_bridge(feature_row, side=side)
            vector = np.asarray(
                [[float(bridge[name]) for name in V8_1_FEATURE_NAMES]],
                dtype=float,
            )
            score = float(booster.predict(xgb.DMatrix(vector, missing=np.nan))[0])
            if not math.isfinite(score):
                raise ValueError("v8.1 transfer prediction is non-finite")
            side_scores[side] = score
        side = max(
            ("UP", "DOWN"),
            key=lambda value: (side_scores[value], value == "UP"),
        )
        score = side_scores[side]
        accepted = score > 0.0 and score >= threshold
        scores.append(score)
        guard_history.append(accepted)
        if not accepted:
            row = _no_trade_row(bundle, policy="v8_1_sell_before_close_bridge")
            row.update(
                {
                    "decision_ts": int(feature_row["decision_ts"]),
                    "selected_side_before_controller": side,
                    "predicted_after_cost_return": score,
                    "controller_threshold": threshold,
                    "controller_quantile": quantile,
                    "controller_band": band,
                    "source_action_score_available": False,
                    "source_action_score_forced_missing": True,
                    "transfer_status": "bridge_handicapped_not_native",
                }
            )
            selected_rows.append(row)
            continue
        features = dict(feature_row["features"])
        decision = {
            "side": side,
            "decision_ts": int(feature_row["decision_ts"]),
            "p_side": float(features[f"{side.lower()}_mid"]),
            "entry_ask": float(features[f"{side.lower()}_ask"]),
            "entry_mid": float(features[f"{side.lower()}_mid"]),
            "predicted_after_cost_return": score,
            "controller_threshold": threshold,
            "controller_quantile": quantile,
            "controller_band": band,
            "source_action_score_available": False,
            "source_action_score_forced_missing": True,
            "transfer_status": "bridge_handicapped_not_native",
        }
        label = _label(
            bundle,
            decision_ts=int(feature_row["decision_ts"]),
            action=f"BUY_{side}_SELL_BEFORE_CLOSE",
        )
        selected_rows.append(
            _cost_row(
                bundle,
                policy="v8_1_sell_before_close_bridge",
                decision=decision,
                label=label,
                predicted_gross_signal=score,
            )
        )
    summary = _policy_summary(selected_rows)
    summary.update(
        {
            "transfer_status": "bridge_handicapped_not_native",
            "native_feature_coverage": {
                "native_feature_count": len(V8_1_NATIVE_FEATURE_NAMES),
                "model_feature_count": len(V8_1_FEATURE_NAMES),
                "coverage": len(V8_1_NATIVE_FEATURE_NAMES) / len(V8_1_FEATURE_NAMES),
                "native_feature_names": list(V8_1_NATIVE_FEATURE_NAMES),
                "forced_missing_feature_names": list(V8_1_FORCED_MISSING_FEATURE_NAMES),
                "bridged_or_assumed_feature_names": list(V8_1_BRIDGED_FEATURE_NAMES),
            },
            "source_action_score_native_coverage": 0.0,
            "source_action_score_missing_was_explicit": True,
            "source_action_score_forced_missing_as_xgboost_nan": True,
            "controller_seed_score_count": int(state.get("eligible_prediction_score_count") or 0),
            "full_retrain_conclusion_allowed": False,
            "promotion_claim_allowed": False,
        }
    )
    return selected_rows, summary


def _v7_feature_bridge(
    feature_row: Mapping[str, Any],
    *,
    side: str,
    run_dir: Path,
) -> tuple[dict[str, Any], int]:
    features = dict(feature_row["features"])
    decision_ts = int(feature_row["decision_ts"])
    prefix = side.lower()
    ask = float(features[f"{prefix}_ask"])
    bid = float(features[f"{prefix}_bid"])
    ask_size = float(features[f"{prefix}_ask_size"])
    bid_size = float(features[f"{prefix}_bid_size"])
    mid = float(features[f"{prefix}_mid"])
    spread = ask - bid
    denominator = bid_size + ask_size
    obi = (bid_size - ask_size) / denominator if denominator > 0 else None
    micro = (ask * bid_size + bid * ask_size) / denominator if denominator > 0 else None
    trades = [
        row
        for row in load_jsonl(run_dir / "provider_raw" / "raw_polymarket_trades.jsonl")
        if str(row.get("outcome") or "").upper() == side
        and decision_ts - 60_000 <= int(row.get("ts") or 0) <= decision_ts
        and int(row.get("available_at_ts") or 0) <= decision_ts
    ]
    signed_volume = sum(
        float(row.get("size") or 0.0)
        * (1.0 if str(row.get("side") or "").upper() == "BUY" else -1.0)
        for row in trades
    )
    trade_volume = sum(float(row.get("size") or 0.0) for row in trades)
    buy_volume = sum(
        float(row.get("size") or 0.0)
        for row in trades
        if str(row.get("side") or "").upper() == "BUY"
    )
    book_rows = [
        row
        for row in load_jsonl(run_dir / "provider_raw" / "raw_polymarket_orderbooks.jsonl")
        if str(row.get("outcome") or "").upper() == side
        and int(row.get("ts") or 0) <= decision_ts
        and int(row.get("available_at_ts") or 0) <= decision_ts
    ]
    ret_1m = _book_return(book_rows, decision_ts, 60_000)
    ret_5m = _book_return(book_rows, decision_ts, 300_000)
    ret_15m = _book_return(book_rows, decision_ts, 900_000)
    rv_1m = _book_volatility(book_rows, decision_ts, 60_000)
    rv_5m = _book_volatility(book_rows, decision_ts, 300_000)
    rv_15m = _book_volatility(book_rows, decision_ts, 900_000)
    ret_30m = _first_finite(ret_15m, ret_5m, ret_1m)
    previous_mid = _latest_book_mid(book_rows, decision_ts - 5_000)
    velocity = None if previous_mid is None else (mid - previous_mid) / 5.0
    last_5s_trade_count = sum(int(row.get("ts") or 0) >= decision_ts - 5_000 for row in trades)
    liquidity_bucket = (
        3.0
        if spread <= 0.02 and trade_volume >= 50.0
        else 2.0
        if spread <= 0.05 and trade_volume >= 10.0
        else 1.0
        if spread <= 0.10
        else 0.0
    )
    dt = datetime.fromtimestamp(decision_ts / 1000, tz=UTC)
    values = {
        "spread": spread,
        "market_implied_prob": ask,
        "mid_price": mid,
        "microprice": micro,
        "obi_l1": obi,
        "obi_l5": None,
        "obi_l10": None,
        "signed_volume_1m": signed_volume,
        "trade_imbalance_1m": (signed_volume / trade_volume if trade_volume > 0 else 0.0),
        "trade_count_1m": float(len(trades)),
        "trade_volume_1m": trade_volume,
        "ret_1m": ret_1m,
        "ret_5m": ret_5m,
        "ret_15m": ret_15m,
        "rv_1m": rv_1m,
        "rv_5m": rv_5m,
        "rv_15m": rv_15m,
        "minute_of_day": (dt.hour * 60 + dt.minute) / 1439.0,
        "day_of_week": float(dt.weekday()),
        "underlying_id": 1.0,
        "horizon_minutes": 15.0,
        "liquidity_bucket": liquidity_bucket,
        "ret_30m": ret_30m,
        "rv_30m": None,
        "aggressor_buy_ratio_1m": (buy_volume / trade_volume if trade_volume > 0 else 0.5),
        "avg_trade_size_1m": (trade_volume / len(trades) if trades else 0.0),
        "tick_spread": spread,
        "tick_obi_l1": obi,
        "tick_obi_l3": None,
        "tick_mid_price": mid,
        "tick_price_velocity": velocity,
        "tick_trade_arrival_rate": last_5s_trade_count / 5.0,
    }
    coverage = sum(_finite_or_none(value) is not None for value in values.values())
    return values, coverage


def _v8_1_feature_bridge(
    feature_row: Mapping[str, Any],
    *,
    side: str,
) -> dict[str, float]:
    features = dict(feature_row["features"])
    prefix = side.lower()
    probability = float(features[f"{prefix}_mid"])
    price = float(features[f"{prefix}_ask"])
    anchors = [
        _finite_or_none(features.get("btc_return_30s")),
        _finite_or_none(features.get("btc_return_1m")),
        _finite_or_none(features.get("reference_price_to_beat_distance_at_decision")),
    ]
    available = [value for value in anchors if value is not None]
    if not available:
        raise ValueError("v8.1 BTC anchor inputs unavailable")
    anchor = statistics.median(available) * (1.0 if side == "UP" else -1.0)
    spread_bps = float(features[f"{prefix}_spread_bps"])
    queue = float(features[f"{prefix}_queue_fill_probability_proxy"])
    staleness = float(features[f"{prefix}_book_staleness_ms"])
    time_to_close = float(features["time_to_close_seconds"])
    horizon_ms = float(feature_row.get("horizon_ms") or 0.0)
    horizon_seconds = horizon_ms / 1000.0
    if not math.isclose(horizon_seconds, 900.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("v8.1 transfer requires the frozen 900-second market horizon")
    return {
        "action_score_available": 0.0,
        "action_score": float("nan"),
        "action_score_margin": float("nan"),
        "btc_anchor_direction": anchor,
        "selected_side_probability": probability,
        "execution_price": price,
        "selected_side_probability_minus_execution_price": probability - price,
        "log1p_spread_bps": math.log1p(max(0.0, spread_bps)),
        "queue_fill_shortfall": 1.0 - min(1.0, max(0.0, queue)),
        "log1p_book_staleness_ms": math.log1p(max(0.0, staleness)),
        "late_window_pressure": max(
            0.0,
            min(1.0, 1.0 - time_to_close / horizon_seconds),
        ),
        "pre_entry_market_exposure": 0.0,
        "same_side_prior_entry": 0.0,
        "side_flip_prior_entry": 0.0,
        "side_is_up": 1.0 if side == "UP" else 0.0,
    }


def _label(
    bundle: Mapping[str, Any],
    *,
    decision_ts: int,
    action: str,
) -> dict[str, Any]:
    matches = [
        row
        for row in bundle["labels"]
        if int(row.get("decision_ts") or 0) == decision_ts
        and str(row.get("action") or "") == action
    ]
    if len(matches) != 1:
        raise ValueError(f"finalized label identity invalid: {bundle['market_id']} {action}")
    return dict(matches[0])


def _cost_row(
    bundle: Mapping[str, Any],
    *,
    policy: str,
    decision: Mapping[str, Any],
    label: Mapping[str, Any],
    predicted_gross_signal: float,
) -> dict[str, Any]:
    entry_mid = float(label["entry_mid"])
    entry_ask = float(label["entry_ask"])
    if str(label["action"]).endswith("HOLD_TO_SETTLEMENT"):
        realized_gross_edge = float(label["settlement_payout"]) - entry_mid
        exit_spread_cost = 0.0
    else:
        exit_bid = float(label["exit_bid"])
        exit_ask = float(label["exit_ask"])
        exit_mid = (exit_bid + exit_ask) / 2.0
        realized_gross_edge = exit_mid - entry_mid
        exit_spread_cost = exit_mid - exit_bid
    entry_spread_cost = entry_ask - entry_mid
    fees = float(label.get("fees") or 0.0)
    slippage = float(label.get("slippage") or 0.0)
    liquidity = float(label.get("liquidity_impact") or 0.0)
    total_cost = entry_spread_cost + exit_spread_cost + fees + slippage + liquidity
    unit_net_pnl = float(label["total_net_pnl_per_notional"])
    reconstructed_unit_net_pnl = realized_gross_edge - total_cost
    reconciliation_error = reconstructed_unit_net_pnl - unit_net_pnl
    if abs(reconciliation_error) > 1e-9:
        raise ValueError(f"transfer cost decomposition does not reconcile: {bundle['market_id']}")
    return {
        "policy": policy,
        "run_id": bundle["run_id"],
        "market_id": bundle["market_id"],
        "accepted": True,
        "side": decision["side"],
        "decision_ts": int(decision["decision_ts"]),
        "resolved_outcome": label["resolved_outcome"],
        "entry_ask": entry_ask,
        "entry_mid": entry_mid,
        "predicted_gross_signal": predicted_gross_signal,
        "realized_gross_edge": realized_gross_edge,
        "entry_spread_cost": entry_spread_cost,
        "exit_spread_cost": exit_spread_cost,
        "fees": fees,
        "slippage": slippage,
        "liquidity_impact": liquidity,
        "total_cost": total_cost,
        "unit_net_pnl": unit_net_pnl,
        "reconstructed_unit_net_pnl": reconstructed_unit_net_pnl,
        "cost_reconciliation_error": reconciliation_error,
        "cost_signal_ratio": (
            total_cost / abs(predicted_gross_signal) if predicted_gross_signal != 0.0 else None
        ),
        "decision": dict(decision),
        "true_paired_executable_ask_used": True,
        "complement_quote_proxy_used": False,
        "position_size": 1.0,
        "development_only_forever": True,
        "promotion_evidence_eligible": False,
        "safety": dict(SAFETY),
    }


def _no_trade_row(
    bundle: Mapping[str, Any],
    *,
    policy: str,
) -> dict[str, Any]:
    return {
        "policy": policy,
        "run_id": bundle["run_id"],
        "market_id": bundle["market_id"],
        "accepted": False,
        "side": "NONE",
        "unit_net_pnl": 0.0,
        "position_size": 0.0,
        "true_paired_executable_ask_used": True,
        "complement_quote_proxy_used": False,
        "development_only_forever": True,
        "promotion_evidence_eligible": False,
        "safety": dict(SAFETY),
    }


def _policy_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    accepted = [row for row in rows if row.get("accepted") is True]
    accepted_unit_net_pnl = [float(row["unit_net_pnl"]) for row in accepted]
    total_cost = sum(float(row["total_cost"]) for row in accepted)
    total_signal = sum(abs(float(row["predicted_gross_signal"])) for row in accepted)
    return {
        "market_count": len(rows),
        "accepted_market_count": len(accepted),
        "acceptance_rate": len(accepted) / len(rows) if rows else 0.0,
        "accepted_up_count": sum(row.get("side") == "UP" for row in accepted),
        "accepted_down_count": sum(row.get("side") == "DOWN" for row in accepted),
        "total_realized_gross_edge": sum(float(row["realized_gross_edge"]) for row in accepted),
        "mean_predicted_gross_signal": _mean(
            [float(row["predicted_gross_signal"]) for row in accepted]
        ),
        "total_entry_spread_cost": sum(float(row["entry_spread_cost"]) for row in accepted),
        "total_exit_spread_cost": sum(float(row["exit_spread_cost"]) for row in accepted),
        "total_fees": sum(float(row["fees"]) for row in accepted),
        "total_slippage": sum(float(row["slippage"]) for row in accepted),
        "total_liquidity_impact": sum(float(row["liquidity_impact"]) for row in accepted),
        "total_cost": total_cost,
        "total_unit_net_pnl": sum(float(row["unit_net_pnl"]) for row in rows),
        "mean_unit_net_pnl_per_accepted_market": _mean(accepted_unit_net_pnl),
        "mean_unit_net_pnl_bootstrap_lcb": _bootstrap_mean_lcb(
            accepted_unit_net_pnl,
            resamples=10000,
            seed=0,
            one_sided_confidence=0.95,
        ),
        "mean_unit_net_pnl_bootstrap": {
            "population": "accepted_markets",
            "resamples": 10000,
            "seed": 0,
            "one_sided_confidence": 0.95,
        },
        "aggregate_cost_signal_ratio": (total_cost / total_signal if total_signal else None),
        "true_paired_executable_ask_coverage": 1.0,
        "complement_quote_proxy_used": False,
        "unit_sizing": True,
    }


def _weak_signal_assessment(
    summary: Mapping[str, Any],
    rule: Mapping[str, Any],
) -> dict[str, Any]:
    accepted = int(summary["accepted_market_count"])
    minimum = int(rule["minimum_accepted_market_count"])
    lcb = _finite_or_none(summary.get("mean_unit_net_pnl_bootstrap_lcb"))
    met = accepted >= minimum and lcb is not None and lcb > 0.0
    return {
        "rule": dict(rule),
        "accepted_market_count": accepted,
        "mean_unit_net_pnl_per_accepted_market": summary[
            "mean_unit_net_pnl_per_accepted_market"
        ],
        "mean_unit_net_pnl_bootstrap_lcb": lcb,
        "weak_signal_rule_met": met,
        "interpretation": (
            "weak_report_only_signal"
            if met
            else "weak_rule_not_met_no_full_retrain_inference"
        ),
        "promotion_claim_made": False,
        "full_retrain_conclusion_made": False,
    }


def _bootstrap_mean_lcb(
    values: Sequence[float],
    *,
    resamples: int,
    seed: int,
    one_sided_confidence: float,
) -> float | None:
    if not values:
        return None
    sample = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.mean(
        rng.choice(sample, size=(resamples, len(sample)), replace=True),
        axis=1,
    )
    return float(np.quantile(means, 1.0 - one_sided_confidence))


def _controller_quantile(history: Sequence[bool]) -> tuple[float, str]:
    window = list(history[-20:])
    if len(window) < 20:
        return 0.4, "initial_q40"
    rate = sum(window) / len(window)
    if rate < 1.0 / 3.0:
        return 0.25, "low_support_q25"
    if rate > 0.5:
        return 0.5, "high_support_q50"
    return 0.4, "balanced_support_q40"


def _finite_sample_quantile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("controller score history unavailable")
    ordered = sorted(float(value) for value in values)
    rank = min(len(ordered), max(1, math.ceil(len(ordered) * quantile)))
    return ordered[rank - 1]


def _book_return(
    rows: Sequence[Mapping[str, Any]],
    decision_ts: int,
    lookback_ms: int,
) -> float | None:
    current = _latest_book_mid(rows, decision_ts)
    previous = _latest_book_mid(rows, decision_ts - lookback_ms)
    if current is None or previous is None or previous == 0.0:
        return None
    return current / previous - 1.0


def _book_volatility(
    rows: Sequence[Mapping[str, Any]],
    decision_ts: int,
    lookback_ms: int,
) -> float | None:
    values = [
        float(row["mid_price"])
        for row in rows
        if decision_ts - lookback_ms <= int(row.get("ts") or 0) <= decision_ts
        and _finite_or_none(row.get("mid_price")) is not None
    ]
    if len(values) < 2:
        return None
    returns = [
        math.log(current / previous)
        for previous, current in zip(values, values[1:], strict=False)
        if previous > 0.0 and current > 0.0
    ]
    return statistics.pstdev(returns) if returns else 0.0


def _latest_book_mid(
    rows: Sequence[Mapping[str, Any]],
    timestamp: int,
) -> float | None:
    eligible = [
        row
        for row in rows
        if int(row.get("ts") or 0) <= timestamp
        and int(row.get("available_at_ts") or 0) <= timestamp
        and _finite_or_none(row.get("mid_price")) is not None
    ]
    if not eligible:
        return None
    return float(max(eligible, key=lambda row: int(row["ts"]))["mid_price"])


def _first_finite(*values: Any) -> float | None:
    for value in values:
        result = _finite_or_none(value)
        if result is not None:
            return result
    return None


def _finite_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _nan_if_missing(value: Any) -> float:
    result = _finite_or_none(value)
    return float("nan") if result is None else result


def _verify_descriptor(descriptor: Mapping[str, Any], label: str) -> None:
    path = _resolve_pinned_path(descriptor.get("path"))
    expected = str(descriptor.get("sha256") or "").lower()
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"{label} bytes do not match pinned SHA-256")


def _resolve_pinned_path(raw_path: Any) -> Path:
    path = Path(str(raw_path or ""))
    if path.is_absolute():
        return path.resolve()
    resolved = (REPO_ROOT / path).resolve()
    if not resolved.is_relative_to(REPO_ROOT):
        raise ValueError("repo-relative pinned path escapes repository root")
    return resolved


def _is_repo_content_addressed_descriptor(descriptor: Mapping[str, Any]) -> bool:
    raw_path = Path(str(descriptor.get("path") or ""))
    expected = str(descriptor.get("sha256") or "").lower()
    parts = raw_path.parts
    if raw_path.is_absolute() or len(expected) != 64:
        return False
    try:
        sha_index = parts.index("sha256")
    except ValueError:
        return False
    return (
        sha_index + 1 < len(parts)
        and parts[sha_index + 1] == expected
        and _resolve_pinned_path(raw_path).is_relative_to(REPO_ROOT)
    )


def _validate_transfer_freeze(freeze: Mapping[str, Any]) -> None:
    if (
        freeze.get("schema_version") != TRANSFER_FREEZE_SCHEMA_VERSION
        or freeze.get("run_count") != 1
        or freeze.get("development_only_forever") is not True
        or freeze.get("promotion_evidence_eligible") is not False
        or freeze.get("safety") != SAFETY
    ):
        raise ValueError("transfer diagnostic freeze semantics invalid")
    for key in (
        "protocol",
        "report",
        "legacy_v7_hold_to_settlement_market_rows",
        "v8_1_sell_before_close_bridge_market_rows",
    ):
        _verify_descriptor(dict(freeze.get(key) or {}), f"transfer freeze {key}")


def _descriptor(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _optional_json(path: Path) -> dict[str, Any]:
    return _load_json(path) if path.is_file() else {}


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    path.write_text(data, encoding="utf-8")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _health_markdown(report: Mapping[str, Any]) -> str:
    cumulative = report["cumulative"]
    pair = cumulative["paired_up_down_executable_ask"]
    causal = cumulative["causal_completeness"]
    return "\n".join(
        [
            "# BTC-15m Development Lane Daily Health",
            "",
            f"- UTC date: `{report['date_utc']}`",
            (
                "- cumulative attempted / quality-valid / finalized-quality-valid: "
                f"`{cumulative['attempted_market_count']} / "
                f"{cumulative['quality_valid_market_count']} / "
                f"{cumulative['quality_valid_outcome_finalized_market_count']}`"
            ),
            (
                "- paired UP+DOWN executable ask coverage: "
                f"`{pair['complete_decision_count']}/{pair['expected_decision_count']}` "
                f"(`{pair['coverage']:.2%}`)"
            ),
            (
                "- causal completeness (book / Chainlink / trade tape): "
                f"`{causal['book']['coverage']:.2%} / "
                f"{causal['chainlink']['coverage']:.2%} / "
                f"{causal['trade_tape']['coverage']:.2%}`"
            ),
            (
                "- exclusion reasons: `"
                + json.dumps(
                    cumulative["exclusion_reason_distribution_multi_label"],
                    sort_keys=True,
                )
                + "`"
            ),
            ("- authorization: pause before attempt 120; `attempt_120_authorized=false`"),
            "",
            "Development data only; never promotion evidence. All safety unlocks remain false.",
            "",
        ]
    )


def _transfer_markdown(report: Mapping[str, Any]) -> str:
    panels = report["policy_panels"]
    v7 = panels["legacy_v7_hold_to_settlement"]["metrics"]
    v81_panel = panels["v8_1_sell_before_close_bridge"]
    v81 = v81_panel["metrics"]
    coverage = v81_panel["native_feature_coverage"]
    weak = v81_panel["weak_signal_rule"]
    v7_ratio = v7["aggregate_cost_signal_ratio"]
    v81_ratio = v81["aggregate_cost_signal_ratio"]
    lines = [
        "# BTC-15m Transfer Diagnostic",
        "",
        "Development diagnostic only; never promotion evidence.",
        "",
        "## Legacy v7 — HOLD_TO_SETTLEMENT",
        "",
        f"- accepted: `{v7['accepted_market_count']}/{v7['market_count']}`",
        f"- UP / DOWN: `{v7['accepted_up_count']} / {v7['accepted_down_count']}`",
        f"- total unit net PnL (report only): `{v7['total_unit_net_pnl']:.6f}`",
        (
            "- cost/signal: `"
            + ("" if v7_ratio is None else f"{v7_ratio:.6f}")
            + "`"
        ),
        "- binary positive-PnL gate: `false`",
        "",
        "## v8.1 — SELL_BEFORE_CLOSE handicapped bridge",
        "",
        "- transfer status: `bridge_handicapped_not_native`",
        (
            "- native feature coverage: "
            f"`{coverage['native_feature_count']}/{coverage['model_feature_count']}` "
            f"(`{coverage['coverage']:.2%}`)"
        ),
        f"- accepted: `{v81['accepted_market_count']}/{v81['market_count']}`",
        f"- UP / DOWN: `{v81['accepted_up_count']} / {v81['accepted_down_count']}`",
        (
            "- mean unit net PnL / bootstrap 95% LCB: "
            f"`{v81['mean_unit_net_pnl_per_accepted_market']} / "
            f"{v81['mean_unit_net_pnl_bootstrap_lcb']}`"
        ),
        (
            "- cost/signal: `"
            + ("" if v81_ratio is None else f"{v81_ratio:.6f}")
            + "`"
        ),
        f"- predeclared weak rule met: `{str(weak['weak_signal_rule_met']).lower()}`",
        f"- interpretation: `{weak['interpretation']}`",
        "- full-retrain conclusion made: `false`",
        "- promotion claim made: `false`",
        "",
        "## Cross-policy boundary",
        "",
        (
            "No cross-policy superiority claim is made: HOLD_TO_SETTLEMENT and "
            "SELL_BEFORE_CLOSE are separate lifecycle panels."
        ),
        "",
    ]
    return "\n".join(lines)


def _readiness_markdown(report: Mapping[str, Any]) -> str:
    metrics = report["metrics"]
    return "\n".join(
        [
            "# BTC-15m Model Training Readiness",
            "",
            f"- training start allowed: `{str(report['training_start_allowed']).lower()}`",
            (
                "- quality-valid outcome-finalized markets: "
                f"`{metrics['quality_valid_outcome_finalized_market_count']}/"
                f"{metrics['required_quality_valid_outcome_finalized_market_count']}`"
            ),
            (
                "- paired executable quote coverage: "
                f"`{metrics['paired_executable_quote_coverage']:.2%}` (minimum `95%`)"
            ),
            ("- blockers: `" + json.dumps(report["blocking_reason_codes"], sort_keys=True) + "`"),
            "",
            (
                "The readiness minimum is reachable under the 119-attempt collector cap. "
                "Attempt 120 is not authorized."
            ),
            "",
        ]
    )
