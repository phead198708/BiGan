"""Frozen future accepted-bet evaluation for the v8 PnL-aligned candidate."""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.contracts import compact_safety_fields
from bigan.v8.polymarket.training.execution_layer_v2_pnl_aligned_action_value import (
    FORBIDDEN_DECISION_FIELDS,
    REQUIRED_ACTIONS,
    _release_closed_shadow_positions,
    build_pnl_aligned_action_conditioned_rows,
    run_pnl_aligned_action_value_outcome_blind_shadow,
    validate_pnl_aligned_action_value_protocol,
)
from bigan.v8.polymarket.training.post_freeze_o_replay_aligned_source_ranking import (
    _v8_apply_simulated_order_to_state,
    _v8_execution_guard_config,
    _v8_execution_guard_decision,
    _v8_initial_runtime_state,
)

EVALUATION_SCHEMA_VERSION = (
    "bigan-v8-execution-layer-v2-pnl-aligned-future-evaluation-protocol-v1"
)
REPORT_SCHEMA_VERSION = (
    "bigan-v8-execution-layer-v2-pnl-aligned-future-accepted-bet-report-v1"
)


@dataclass(frozen=True, slots=True)
class PnLAlignedFutureEvaluationFreezeConfig:
    """Inputs frozen before future settlement targets are reconciled."""

    run_id: str
    output_dir: Path | str
    evaluation_protocol_path: Path | str
    expected_evaluation_protocol_sha256: str
    collection_freeze_manifest_path: Path | str
    expected_collection_freeze_manifest_sha256: str
    model_dir: Path | str
    git_commit: str

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        for name, value in (
            ("expected_evaluation_protocol_sha256", self.expected_evaluation_protocol_sha256),
            (
                "expected_collection_freeze_manifest_sha256",
                self.expected_collection_freeze_manifest_sha256,
            ),
        ):
            if not _is_sha256(value):
                raise ValueError(f"{name} must be a SHA-256 digest")
        if len(self.git_commit) != 40 or any(
            char not in "0123456789abcdef" for char in self.git_commit.lower()
        ):
            raise ValueError("git_commit must be a 40-character hex digest")
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(
            self, "evaluation_protocol_path", Path(self.evaluation_protocol_path)
        )
        object.__setattr__(
            self,
            "collection_freeze_manifest_path",
            Path(self.collection_freeze_manifest_path),
        )
        object.__setattr__(self, "model_dir", Path(self.model_dir))


def validate_pnl_aligned_future_evaluation_protocol(
    protocol: dict[str, Any],
) -> None:
    """Reject metric, baseline, tuning, or safety drift."""

    bootstrap = dict(protocol.get("market_bootstrap") or {})
    gates = dict(protocol.get("future_evidence_gates") or {})
    safety = dict(protocol.get("safety") or {})
    checks = {
        "schema": protocol.get("schema_version") == EVALUATION_SCHEMA_VERSION,
        "frozen": protocol.get("frozen") is True,
        "diagnostic_only": protocol.get("diagnostic_only") is True,
        "candidate": protocol.get("candidate_policy_name")
        == "pnl_aligned_action_conditioned_net_value_v1",
        "baseline": protocol.get("baseline_policy_name")
        == "raw_market_probability_selected_o_action_baseline",
        "baseline_action_source": protocol.get("baseline_action_source")
        == "canonical_o_rank_1_action",
        "baseline_not_calibrated_fair_value": protocol.get(
            "baseline_market_probability_is_calibrated_fair_value"
        )
        is False,
        "cost_rule": protocol.get("decision_time_execution_cost_rule_id")
        == "spread_queue_staleness_cost_proxy_v1",
        "threshold": float(protocol.get("frozen_entry_edge_threshold") or -1.0)
        == 0.02,
        "bootstrap": (
            int(bootstrap.get("resample_count") or 0) == 2000
            and int(bootstrap.get("seed") or 0) == 20260715
            and float(bootstrap.get("confidence_level") or 0.0) == 0.95
            and bootstrap.get("sampling_unit") == "market_id"
        ),
        "support": (
            int(gates.get("minimum_unique_market_count") or 0) == 30
            and int(gates.get("minimum_accepted_bet_count") or 0) == 30
            and int(gates.get("minimum_accepted_bet_count_per_side") or 0) == 10
            and gates.get("all_accepted_bets_must_be_settled") is True
        ),
        "outcomes_not_selection": protocol.get("outcome_fields_used_for_shadow_selection")
        is False,
        "outcomes_evaluation_only": protocol.get("outcome_fields_used_for_evaluation_only")
        is True,
        "no_future_feature_tuning": protocol.get(
            "uses_future_outcomes_for_feature_selection"
        )
        is False,
        "no_future_hyperparameter_tuning": protocol.get(
            "uses_future_outcomes_for_hyperparameter_selection"
        )
        is False,
        "no_future_threshold_tuning": protocol.get(
            "uses_future_outcomes_for_threshold_selection"
        )
        is False,
        "no_future_guard_tuning": protocol.get(
            "uses_future_outcomes_for_guard_or_sizing_selection"
        )
        is False,
        "no_source_score_mutation": protocol.get("source_o_score_mutation_allowed")
        is False,
        "no_source_ranking_mutation": protocol.get("source_ranking_mutation_allowed")
        is False,
        "safety": (
            safety.get("paper_only") is True
            and safety.get("capital_at_risk") is False
            and safety.get("polymarket_write_enabled") is False
            and safety.get("wallet_signing_enabled") is False
            and safety.get("source_model_candidate_eligible") is False
            and safety.get("freeze_ready") is False
            and safety.get("promotion_evidence_eligible") is False
            and safety.get("v8_execution_handoff_allowed") is False
            and safety.get("#134_resume_allowed") is False
            and safety.get("#146_start_allowed") is False
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError("invalid future evaluation protocol: " + ", ".join(failed))


def freeze_pnl_aligned_future_evaluation(
    config: PnLAlignedFutureEvaluationFreezeConfig,
) -> dict[str, Any]:
    """Freeze all evaluator inputs and metrics before outcome reconciliation."""

    protocol_path = config.evaluation_protocol_path.resolve()
    collection_freeze_path = config.collection_freeze_manifest_path.resolve()
    if _sha256_file(protocol_path) != config.expected_evaluation_protocol_sha256:
        raise ValueError("future evaluation protocol SHA-256 mismatch")
    if (
        _sha256_file(collection_freeze_path)
        != config.expected_collection_freeze_manifest_sha256
    ):
        raise ValueError("collection freeze manifest SHA-256 mismatch")
    protocol = _load_json(protocol_path)
    validate_pnl_aligned_future_evaluation_protocol(protocol)
    collection_freeze = _load_json(collection_freeze_path)
    model_dir = config.model_dir.resolve()
    fit_manifest_path = model_dir / "pnl_aligned_action_value_fit_manifest.json"
    fit_manifest = _load_json(fit_manifest_path)
    for name, descriptor in (
        ("model", fit_manifest.get("model")),
        ("protocol", fit_manifest.get("protocol")),
    ):
        verified = _verified_descriptor(descriptor, name=name)
        if collection_freeze.get(name) != verified:
            raise ValueError(f"collection freeze {name} lineage mismatch")
    guard_config = _v8_execution_guard_config()
    if collection_freeze.get("execution_guard_config_sha256") != canonical_json_sha256(
        guard_config
    ):
        raise ValueError("collection freeze execution guard hash mismatch")
    if collection_freeze.get("model_config_or_threshold_mutation_after_freeze_allowed") is not False:
        raise ValueError("collection freeze mutation policy is not fail closed")

    output_dir = config.output_dir / config.run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": (
            "bigan-v8-execution-layer-v2-pnl-aligned-future-evaluation-freeze-v1"
        ),
        "run_id": config.run_id,
        "freeze_created_ts": int(time.time() * 1000),
        "git_commit": config.git_commit.lower(),
        "evaluation_protocol": _descriptor(protocol_path),
        "collection_freeze_manifest": _descriptor(collection_freeze_path),
        "collection_freeze_id": collection_freeze["collection_freeze_id"],
        "model": fit_manifest["model"],
        "model_contract": fit_manifest["model_contract"],
        "model_protocol": fit_manifest["protocol"],
        "execution_guard_config": guard_config,
        "execution_guard_config_sha256": canonical_json_sha256(guard_config),
        "minimum_future_window_start_ts": collection_freeze[
            "minimum_future_window_start_ts"
        ],
        "prior_market_ids_sha256": collection_freeze["prior_market_ids_sha256"],
        "future_outcome_targets_loaded": False,
        "shadow_decisions_generated": False,
        "outcome_reconciliation_started": False,
        "exactly_once_evaluation_required": True,
        "future_results_may_mutate_protocol_model_threshold_guard_or_sizing": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "v8_execution_handoff_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        **compact_safety_fields(),
    }
    manifest["evaluation_freeze_id"] = canonical_json_sha256(manifest)
    manifest_path = output_dir / "pnl_aligned_future_evaluation_freeze_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "output_dir": output_dir,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
        "manifest": manifest,
    }


def run_pnl_aligned_future_outcome_blind_shadow_comparison(
    *,
    model_dir: Path | str,
    decision_rows: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Emit candidate and baseline shadows without reading future targets."""

    candidate_rows, candidate_report = run_pnl_aligned_action_value_outcome_blind_shadow(
        model_dir=model_dir,
        decision_rows=decision_rows,
    )
    baseline_rows, baseline_report = _run_outcome_blind_raw_probability_baseline(
        model_dir=model_dir,
        decision_rows=decision_rows,
    )
    candidate_ids = {str(row["source_row_identity"]) for row in candidate_rows}
    baseline_ids = {str(row["source_row_identity"]) for row in baseline_rows}
    identity_match = candidate_ids == baseline_ids and len(candidate_ids) == len(
        decision_rows
    )
    status = (
        "OUTCOME_BLIND_COMPARISON_SHADOW_COMPLETE"
        if identity_match
        and candidate_report.get("status") == "OUTCOME_BLIND_SHADOW_EXECUTION_COMPLETE"
        and baseline_report.get("status") == "OUTCOME_BLIND_BASELINE_COMPLETE"
        else "BLOCKED_FAIL_CLOSED"
    )
    report = {
        "schema_version": (
            "bigan-v8-execution-layer-v2-pnl-aligned-future-shadow-comparison-v1"
        ),
        "status": status,
        "decision_count": len(decision_rows),
        "candidate_shadow_report": candidate_report,
        "baseline_shadow_report": baseline_report,
        "candidate_baseline_identity_match": identity_match,
        "future_outcome_targets_loaded": False,
        "outcome_fields_used_for_selection": False,
        "source_o_score_mutated": False,
        "source_ranking_mutated": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "v8_execution_handoff_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        **compact_safety_fields(),
    }
    return {"candidate": candidate_rows, "baseline": baseline_rows}, report


def evaluate_pnl_aligned_future_accepted_bets(
    *,
    evaluation_protocol: dict[str, Any],
    collection_freeze_manifest: dict[str, Any],
    candidate_shadow_rows: list[dict[str, Any]],
    baseline_shadow_rows: list[dict[str, Any]],
    settled_evaluation_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Reconcile frozen shadows to post-close targets exactly once."""

    validate_pnl_aligned_future_evaluation_protocol(evaluation_protocol)
    prior_rows_path = Path(
        str(collection_freeze_manifest["historical_development_rows"]["path"])
    )
    if _sha256_file(prior_rows_path) != collection_freeze_manifest[
        "historical_development_rows"
    ]["sha256"]:
        raise ValueError("historical lineage rows descriptor mismatch")
    prior_market_ids = {
        str(row["market_id"]) for row in _load_jsonl(prior_rows_path)
    }
    targets_by_identity = _settled_targets_by_identity(settled_evaluation_rows)
    candidate_ids = {str(row["source_row_identity"]) for row in candidate_shadow_rows}
    baseline_ids = {str(row["source_row_identity"]) for row in baseline_shadow_rows}
    target_ids = set(targets_by_identity)
    identity_match = candidate_ids == baseline_ids == target_ids
    all_shadow_rows = [*candidate_shadow_rows, *baseline_shadow_rows]
    future_boundary_violations = [
        str(row["source_row_identity"])
        for row in all_shadow_rows
        if int(row["decision_ts"])
        < int(collection_freeze_manifest["minimum_future_window_start_ts"])
    ]
    overlapping_markets = sorted(
        {str(row["market_id"]) for row in all_shadow_rows} & prior_market_ids
    )
    shadow_forbidden = sorted(
        {
            field
            for row in all_shadow_rows
            for field in _find_forbidden_fields(row)
        }
    )
    if not identity_match:
        raise ValueError("candidate, baseline, and target identities do not match")
    if future_boundary_violations:
        raise ValueError("future window contains non-future decision timestamps")
    if overlapping_markets:
        raise ValueError("future markets overlap historical fit markets")
    if shadow_forbidden:
        raise ValueError("outcome-blind shadows contain forbidden fields")

    pnl_rows: list[dict[str, Any]] = []
    policy_metrics: dict[str, dict[str, Any]] = {}
    policy_rows = {
        str(evaluation_protocol["candidate_policy_name"]): candidate_shadow_rows,
        str(evaluation_protocol["baseline_policy_name"]): baseline_shadow_rows,
    }
    for policy_name, shadow_rows in policy_rows.items():
        reconciled = [
            _reconcile_shadow_row(
                policy_name=policy_name,
                shadow_row=row,
                target_row=targets_by_identity[str(row["source_row_identity"])],
            )
            for row in shadow_rows
        ]
        pnl_rows.extend(reconciled)
        policy_metrics[policy_name] = _accepted_bet_metrics(reconciled)

    candidate_name = str(evaluation_protocol["candidate_policy_name"])
    baseline_name = str(evaluation_protocol["baseline_policy_name"])
    candidate_metrics = policy_metrics[candidate_name]
    baseline_metrics = policy_metrics[baseline_name]
    market_delta = _market_pnl_delta(
        candidate_rows=[row for row in pnl_rows if row["policy_name"] == candidate_name],
        baseline_rows=[row for row in pnl_rows if row["policy_name"] == baseline_name],
    )
    bootstrap = _market_bootstrap_interval(
        market_delta,
        protocol=evaluation_protocol,
    )
    gates = dict(evaluation_protocol["future_evidence_gates"])
    checks = {
        "minimum_unique_market_count_met": candidate_metrics[
            "accepted_unique_market_count"
        ]
        >= int(gates["minimum_unique_market_count"]),
        "minimum_accepted_bet_count_met": candidate_metrics["accepted_bet_count"]
        >= int(gates["minimum_accepted_bet_count"]),
        "minimum_up_accepted_bet_count_met": candidate_metrics[
            "accepted_bet_count_by_side"
        ].get("UP", 0)
        >= int(gates["minimum_accepted_bet_count_per_side"]),
        "minimum_down_accepted_bet_count_met": candidate_metrics[
            "accepted_bet_count_by_side"
        ].get("DOWN", 0)
        >= int(gates["minimum_accepted_bet_count_per_side"]),
        "all_accepted_bets_settled": candidate_metrics["unresolved_accepted_bet_count"]
        == 0,
        "candidate_net_pnl_positive": candidate_metrics["settled_net_pnl_sum"] > 0.0,
        "candidate_roi_positive": candidate_metrics["roi"] > 0.0,
        "candidate_net_pnl_exceeds_baseline": candidate_metrics[
            "settled_net_pnl_sum"
        ]
        > baseline_metrics["settled_net_pnl_sum"],
        "market_bootstrap_interval_reported": bootstrap["reported"],
        "largest_winner_removal_reported": candidate_metrics[
            "largest_winner_removal"
        ]["reported"],
        "zero_forbidden_shadow_fields": not shadow_forbidden,
        "future_window_strictly_later": not future_boundary_violations,
        "future_markets_disjoint": not overlapping_markets,
    }
    reason_map = {
        "minimum_unique_market_count_met": "insufficient_unique_market_support",
        "minimum_accepted_bet_count_met": "insufficient_accepted_bet_support",
        "minimum_up_accepted_bet_count_met": "insufficient_up_accepted_bet_support",
        "minimum_down_accepted_bet_count_met": "insufficient_down_accepted_bet_support",
        "all_accepted_bets_settled": "unresolved_accepted_bets",
        "candidate_net_pnl_positive": "candidate_net_pnl_not_positive",
        "candidate_roi_positive": "candidate_roi_not_positive",
        "candidate_net_pnl_exceeds_baseline": "candidate_not_better_than_baseline",
        "market_bootstrap_interval_reported": "market_bootstrap_not_reported",
        "largest_winner_removal_reported": "largest_winner_removal_not_reported",
        "zero_forbidden_shadow_fields": "forbidden_shadow_fields_present",
        "future_window_strictly_later": "future_window_not_strictly_later",
        "future_markets_disjoint": "future_markets_overlap_historical_fit",
    }
    blockers = [reason_map[name] for name, passed in checks.items() if not passed]
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "FUTURE_ACCEPTED_BET_EVALUATION_COMPLETE",
        "candidate_policy_name": candidate_name,
        "baseline_policy_name": baseline_name,
        "identity_reconciliation_passed": identity_match,
        "future_window_time_validation_passed": not future_boundary_violations,
        "future_market_disjointness_passed": not overlapping_markets,
        "forbidden_shadow_field_violation_count": len(shadow_forbidden),
        "candidate_policy_metrics": candidate_metrics,
        "baseline_policy_metrics": baseline_metrics,
        "candidate_minus_baseline_net_pnl": candidate_metrics[
            "settled_net_pnl_sum"
        ]
        - baseline_metrics["settled_net_pnl_sum"],
        "market_level_candidate_minus_baseline_pnl": market_delta,
        "market_bootstrap_interval": bootstrap,
        "future_evidence_gate_checks": checks,
        "future_evidence_gate_passed": all(checks.values()),
        "future_evidence_gate_blocking_reason_codes": blockers,
        "future_results_used_for_tuning": False,
        "future_results_used_for_unlock": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "v8_execution_handoff_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        **compact_safety_fields(),
    }
    return report, sorted(
        pnl_rows,
        key=lambda row: (
            int(row["decision_ts"]),
            str(row["market_id"]),
            str(row["policy_name"]),
        ),
    )


def _run_outcome_blind_raw_probability_baseline(
    *,
    model_dir: Path | str,
    decision_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model_dir = Path(model_dir).resolve()
    fit_manifest = _load_json(model_dir / "pnl_aligned_action_value_fit_manifest.json")
    protocol = _load_json(Path(fit_manifest["protocol"]["path"]))
    validate_pnl_aligned_action_value_protocol(protocol)
    action_rows, audit = build_pnl_aligned_action_conditioned_rows(
        decision_rows,
        protocol=protocol,
        require_targets=False,
    )
    if audit["blocking_reason_codes"]:
        return [], {
            "status": "BLOCKED_FAIL_CLOSED",
            "feature_leakage_audit": audit,
            "source_model_candidate_eligible": False,
            **compact_safety_fields(),
        }
    threshold = float(protocol["frozen_execution_contract"]["entry_edge_threshold"])
    guard_config = _v8_execution_guard_config()
    state = _v8_initial_runtime_state(guard_config)
    market_close_by_open_position: dict[str, int] = {}
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in action_rows:
        grouped[(str(row["market_id"]), int(row["decision_ts"]))].append(row)
    replay_rows: list[dict[str, Any]] = []
    for index, ((market_id, decision_ts), rows) in enumerate(
        sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])), start=1
    ):
        _release_closed_shadow_positions(
            state=state,
            market_close_by_open_position=market_close_by_open_position,
            decision_ts=decision_ts,
        )
        selected = min(
            rows,
            key=lambda row: (
                float(row["decision_time_features"]["canonical_action_rank"]),
                str(row["action"]),
            ),
        )
        handoff = dict(selected["execution_handoff_context"])
        side = str(selected["side"])
        probability = (
            float(handoff["p_up"])
            if side == "UP"
            else float(handoff["p_down"])
            if side == "DOWN"
            else 0.0
        )
        micro = dict(handoff.get("microstructure_snapshot") or {})
        execution_price = float(micro.get("entry_ask") or 0.0)
        cost = _decision_time_execution_cost(micro)
        edge = probability - execution_price - cost
        selected_action = str(selected["action"])
        signal_passed = selected_action != "NO_TRADE" and edge >= threshold
        blockers: list[str] = []
        guard_row: dict[str, Any] | None = None
        if selected_action == "NO_TRADE":
            blockers.append("raw_baseline_selected_no_trade")
        elif edge < threshold:
            blockers.append("raw_baseline_edge_below_frozen_threshold")
        else:
            guard_row = _v8_execution_guard_decision(
                handoff,
                guard_config=guard_config,
                runtime_state=state,
                runtime_mode="simulated_runtime_state",
            )
            blockers.extend(guard_row["execution_blocking_reason_codes"])
        guard_allowed = bool(guard_row and guard_row["order_allowed"])
        order_id = None
        if guard_allowed:
            order_id = f"raw-probability-baseline-bet-{index:06d}"
            _v8_apply_simulated_order_to_state(
                state=state,
                decision=guard_row,
                simulated_order_id=order_id,
            )
            market_close_by_open_position[market_id] = int(selected["market_close_ts"])
        replay_row = {
            "policy_name": "raw_market_probability_selected_o_action_baseline",
            "source_row_identity": str(selected["source_row_identity"]),
            "market_id": market_id,
            "decision_ts": decision_ts,
            "market_close_ts": int(selected["market_close_ts"]),
            "selected_action": selected_action,
            "selected_side": side,
            "selected_action_family": str(selected["action_family"]),
            "selected_side_market_probability": probability,
            "selected_execution_price": execution_price,
            "decision_time_expected_execution_cost_per_unit": cost,
            "model_entry_edge": edge,
            "frozen_entry_edge_threshold": threshold,
            "model_signal_passed": signal_passed,
            "execution_guard_evaluated": guard_row is not None,
            "execution_guard_order_allowed": guard_allowed,
            "execution_guarded_action": (
                guard_row.get("execution_guarded_action") if guard_row else None
            ),
            "execution_guarded_side": (
                guard_row.get("execution_guarded_side") if guard_row else None
            ),
            "proposed_order_size": (
                float(guard_row["proposed_order_size"]) if guard_allowed else 0.0
            ),
            "simulated_order_id": order_id,
            "execution_blocking_reason_codes": sorted(set(blockers)),
            "execution_guard_reason_codes": (
                list(guard_row["execution_guard_reason_codes"]) if guard_row else []
            ),
            "outcome_fields_used": False,
            "realized_pnl_used": False,
            "source_o_score_mutated": False,
            "source_ranking_mutated": False,
            "paper_only": True,
            "capital_at_risk": False,
            "polymarket_write_enabled": False,
            "wallet_signing_enabled": False,
        }
        replay_row["shadow_replay_row_sha256"] = canonical_json_sha256(replay_row)
        replay_rows.append(replay_row)
    accepted = [row for row in replay_rows if row["execution_guard_order_allowed"]]
    report = {
        "status": "OUTCOME_BLIND_BASELINE_COMPLETE",
        "decision_count": len(replay_rows),
        "model_trade_candidate_count": sum(row["model_signal_passed"] for row in replay_rows),
        "executable_shadow_bet_count": len(accepted),
        "feature_leakage_audit": audit,
        "market_probability_is_diagnostic_baseline_not_calibrated_fair_value": True,
        "outcome_fields_used": False,
        "source_o_score_mutated": False,
        "source_ranking_mutated": False,
        "source_model_candidate_eligible": False,
        "promotion_evidence_eligible": False,
        **compact_safety_fields(),
    }
    return replay_rows, report


def _decision_time_execution_cost(micro: dict[str, Any]) -> float:
    spread = max(float(micro.get("spread_bps") or 0.0), 0.0) / 20000.0
    queue = min(max(float(micro.get("queue_fill_proxy") or 0.0), 0.0), 1.0)
    staleness = max(float(micro.get("book_staleness_ms") or 0.0), 0.0)
    return min(
        0.05,
        0.001
        + spread
        + (1.0 - queue) * 0.002
        + min(staleness / 1000.0, 1.0) * 0.001,
    )


def _settled_targets_by_identity(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        identity = str(row.get("row_identity") or "")
        if not identity or identity in result:
            raise ValueError("settled evaluation row identity missing or duplicated")
        targets = dict(row.get("evaluation_target_net_pnl_per_contract_by_action") or {})
        components = dict(row.get("evaluation_target_pnl_components_by_action") or {})
        if set(targets) != set(REQUIRED_ACTIONS) or set(components) != set(
            REQUIRED_ACTIONS
        ):
            raise ValueError("settled evaluation action target grid is incomplete")
        result[identity] = row
    return result


def _reconcile_shadow_row(
    *,
    policy_name: str,
    shadow_row: dict[str, Any],
    target_row: dict[str, Any],
) -> dict[str, Any]:
    if (
        str(shadow_row["market_id"]) != str(target_row["market_id"])
        or int(shadow_row["decision_ts"]) != int(target_row["decision_ts"])
    ):
        raise ValueError("shadow and target row provenance mismatch")
    accepted = shadow_row.get("execution_guard_order_allowed") is True
    action = str(shadow_row.get("execution_guarded_action") or "")
    targets = dict(target_row["evaluation_target_net_pnl_per_contract_by_action"])
    components = dict(target_row["evaluation_target_pnl_components_by_action"])
    target = targets.get(action) if accepted else None
    component = dict(components.get(action) or {}) if accepted else {}
    required_components = (
        "gross_pnl_per_contract",
        "execution_cost_per_contract",
        "net_pnl_per_contract",
    )
    settled = accepted and _finite(target) and all(
        _finite(component.get(name)) for name in required_components
    )
    size = float(shadow_row.get("proposed_order_size") or 0.0) if accepted else 0.0
    entry_price = float(shadow_row.get("selected_execution_price") or 0.0)
    gross_pnl = size * float(component["gross_pnl_per_contract"]) if settled else None
    execution_cost = (
        size * float(component["execution_cost_per_contract"]) if settled else None
    )
    net_pnl = size * float(target) if settled else None
    cost_basis = (
        size * (entry_price + float(component["execution_cost_per_contract"]))
        if settled
        else 0.0
    )
    return {
        "policy_name": policy_name,
        "source_row_identity": str(shadow_row["source_row_identity"]),
        "market_id": str(shadow_row["market_id"]),
        "decision_ts": int(shadow_row["decision_ts"]),
        "market_close_ts": int(shadow_row["market_close_ts"]),
        "selected_action": str(shadow_row["selected_action"]),
        "execution_guarded_action": action or None,
        "execution_guarded_side": shadow_row.get("execution_guarded_side"),
        "execution_guard_order_allowed": accepted,
        "simulated_order_id": shadow_row.get("simulated_order_id"),
        "paper_bet_contract_size": size,
        "execution_price": entry_price,
        "settlement_target_available": settled,
        "guarded_action_target_net_pnl_per_contract": float(target)
        if settled
        else None,
        "gross_pnl": gross_pnl,
        "execution_cost": execution_cost,
        "cost_basis": cost_basis,
        "settled_net_pnl": net_pnl,
        "selection_uses_outcome_fields": False,
        "outcome_aware_evaluation_only": True,
        "source_o_score_mutated": False,
        "source_ranking_mutated": False,
        "paper_only": True,
        "capital_at_risk": False,
    }


def _accepted_bet_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [row for row in rows if row["execution_guard_order_allowed"]]
    settled = [row for row in accepted if row["settlement_target_available"]]
    pnl_values = [float(row["settled_net_pnl"]) for row in settled]
    cost_basis = sum(float(row["cost_basis"]) for row in settled)
    net_pnl = sum(pnl_values)
    market_pnl: dict[str, float] = defaultdict(float)
    for row in settled:
        market_pnl[str(row["market_id"])] += float(row["settled_net_pnl"])
    ordered = sorted(
        settled,
        key=lambda row: (
            int(row["market_close_ts"]),
            str(row["market_id"]),
            str(row["simulated_order_id"]),
        ),
    )
    equity = peak = max_drawdown = 0.0
    for row in ordered:
        equity += float(row["settled_net_pnl"])
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    largest_market = max(market_pnl, key=market_pnl.get) if market_pnl else None
    largest_value = market_pnl.get(largest_market, 0.0) if largest_market else 0.0
    side_counts = Counter(str(row["execution_guarded_side"]) for row in accepted)
    action_counts = Counter(str(row["execution_guarded_action"]) for row in accepted)
    family_counts = Counter(_family(str(row["execution_guarded_action"])) for row in accepted)
    return {
        "accepted_bet_count": len(accepted),
        "settled_accepted_bet_count": len(settled),
        "unresolved_accepted_bet_count": len(accepted) - len(settled),
        "accepted_unique_market_count": len({row["market_id"] for row in accepted}),
        "accepted_bet_count_by_side": dict(sorted(side_counts.items())),
        "accepted_bet_count_by_action": dict(sorted(action_counts.items())),
        "accepted_bet_count_by_family": dict(sorted(family_counts.items())),
        "contract_size_sum": sum(float(row["paper_bet_contract_size"]) for row in accepted),
        "gross_pnl_sum": sum(float(row["gross_pnl"]) for row in settled),
        "execution_cost_sum": sum(float(row["execution_cost"]) for row in settled),
        "cost_basis_sum": cost_basis,
        "settled_net_pnl_sum": net_pnl,
        "roi": net_pnl / cost_basis if cost_basis > 0.0 else 0.0,
        "win_rate": sum(value > 0.0 for value in pnl_values) / len(pnl_values)
        if pnl_values
        else 0.0,
        "chronological_max_drawdown": max_drawdown,
        "pnl_by_market": dict(sorted(market_pnl.items())),
        "largest_winner_removal": {
            "reported": True,
            "largest_winning_market_id": largest_market,
            "largest_winning_market_pnl": largest_value,
            "net_pnl_after_largest_winner_removed": net_pnl - max(largest_value, 0.0),
        },
    }


def _market_pnl_delta(
    *,
    candidate_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
        values: dict[str, float] = defaultdict(float)
        for row in rows:
            if row["settlement_target_available"]:
                values[str(row["market_id"])] += float(row["settled_net_pnl"])
        return values

    candidate = aggregate(candidate_rows)
    baseline = aggregate(baseline_rows)
    return [
        {
            "market_id": market_id,
            "candidate_net_pnl": candidate.get(market_id, 0.0),
            "baseline_net_pnl": baseline.get(market_id, 0.0),
            "candidate_minus_baseline_net_pnl": candidate.get(market_id, 0.0)
            - baseline.get(market_id, 0.0),
        }
        for market_id in sorted(set(candidate) | set(baseline))
    ]


def _market_bootstrap_interval(
    rows: list[dict[str, Any]],
    *,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    config = dict(protocol["market_bootstrap"])
    values = [float(row["candidate_minus_baseline_net_pnl"]) for row in rows]
    if not values:
        return {
            "reported": True,
            "market_count": 0,
            "point_estimate": 0.0,
            "lower_bound": 0.0,
            "upper_bound": 0.0,
            **config,
        }
    rng = random.Random(int(config["seed"]))
    sample_count = int(config["resample_count"])
    estimates = [
        sum(rng.choice(values) for _ in values) for _ in range(sample_count)
    ]
    alpha = (1.0 - float(config["confidence_level"])) / 2.0
    return {
        "reported": True,
        "market_count": len(values),
        "point_estimate": sum(values),
        "lower_bound": float(np.quantile(estimates, alpha)),
        "upper_bound": float(np.quantile(estimates, 1.0 - alpha)),
        **config,
    }


def _family(action: str) -> str:
    if action.endswith("HOLD_TO_SETTLEMENT"):
        return "HOLD_TO_SETTLEMENT"
    if action.endswith("SELL_BEFORE_CLOSE"):
        return "SELL_BEFORE_CLOSE"
    return "NO_TRADE"


def _find_forbidden_fields(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized = str(key).lower()
            if normalized in FORBIDDEN_DECISION_FIELDS or any(
                token in normalized
                for token in (
                    "resolved_outcome",
                    "settlement_pnl",
                    "future_return",
                    "oracle_action",
                    "target_net_pnl",
                )
            ):
                found.add(str(key))
            found.update(_find_forbidden_fields(nested))
    elif isinstance(value, list):
        for nested in value:
            found.update(_find_forbidden_fields(nested))
    return found


def _verified_descriptor(value: Any, *, name: str) -> dict[str, str]:
    descriptor = dict(value or {})
    path = Path(str(descriptor.get("path") or ""))
    if not path.is_file() or descriptor.get("sha256") != _sha256_file(path):
        raise ValueError(f"{name} descriptor hash mismatch")
    return {"path": str(path), "sha256": str(descriptor["sha256"])}


def _descriptor(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256_file(path)}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        char in "0123456789abcdef" for char in value.lower()
    )


def _finite(value: Any) -> bool:
    return isinstance(value, int | float) and math.isfinite(float(value))
