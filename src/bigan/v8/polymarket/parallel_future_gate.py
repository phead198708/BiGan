"""Outcome-blind parallel future gate for v8.1, v8.3, and matched v6.7."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any

from bigan.v8.canonical_payload import canonical_payload_sha256

PARALLEL_PROTOCOL_SCHEMA_VERSION = "bigan-v8-parallel-candidate-protocol-v1"
PARALLEL_COLLECTION_PLAN_SCHEMA_VERSION = (
    "bigan-v8-parallel-future-collection-plan-v1"
)
PARALLEL_FREEZE_SCHEMA_VERSION = "bigan-v8-parallel-target-free-freeze-v1"
PARALLEL_EVALUATION_SCHEMA_VERSION = "bigan-v8-parallel-future-evaluation-v1"
DECISION_STREAM_SCHEMA_VERSION = "bigan-v8-parallel-candidate-decision-stream-v1"
FROZEN_MODEL_BINDING_SCHEMA_VERSION = (
    "bigan-v8-parallel-frozen-model-binding-v1"
)
TARGET_FIELDS = frozenset(
    {
        "outcome",
        "resolved_outcome",
        "settlement",
        "settlement_payout",
        "label",
        "target",
        "future_return",
        "return",
        "pnl",
        "profit",
        "winner",
        "oracle_action",
    }
)
REQUIRED_CANDIDATES = (
    "v8_1_primary_no_fallback",
    "v8_3_primary_with_fallback",
    "matched_frozen_v6_7",
)


class ParallelFutureGateError(ValueError):
    """Raised when the parallel future gate cannot remain outcome-blind."""


def validate_parallel_future_collection_plan(
    plan: dict[str, Any],
    *,
    protocol_sha256: str,
    candidate_contract_sha256s: dict[str, str],
    collector_protocol_sha256: str,
    feature_contract_sha256: str,
    frozen_model_binding_sha256: str,
    frozen_model_binding: dict[str, Any],
    historical_gate_contract_sha256: str,
    historical_replay_report_sha256: str,
    historical_replay_report: dict[str, Any],
    collection_started_ts: int | None = None,
) -> None:
    """Validate the immutable, pre-collection instance of the parallel gate."""

    blockers: list[str] = []
    freeze_created_ts = int(plan.get("freeze_created_ts") or 0)
    if plan.get("schema_version") != PARALLEL_COLLECTION_PLAN_SCHEMA_VERSION:
        blockers.append("schema_version")
    if plan.get("frozen") is not True:
        blockers.append("not_frozen")
    if plan.get("preregistered_before_collection") is not True:
        blockers.append("not_preregistered")
    if freeze_created_ts <= 0:
        blockers.append("freeze_created_ts")
    if (
        collection_started_ts is not None
        and collection_started_ts <= freeze_created_ts
    ):
        blockers.append("collection_not_strictly_later")

    lineage = dict(plan.get("lineage") or {})
    implementation_commit = str(lineage.get("implementation_commit") or "")
    if len(implementation_commit) != 40 or any(
        character not in "0123456789abcdef" for character in implementation_commit
    ):
        blockers.append("implementation_commit")
    expected_lineage = {
        "parallel_candidate_protocol_sha256": protocol_sha256,
        "persistent_collector_protocol_sha256": collector_protocol_sha256,
        "feature_contract_sha256": feature_contract_sha256,
        "frozen_model_binding_sha256": frozen_model_binding_sha256,
    }
    for name, expected in expected_lineage.items():
        if str(lineage.get(name) or "").lower() != expected.lower():
            blockers.append(name)
    frozen_candidate_hashes = dict(
        lineage.get("candidate_contract_sha256s") or {}
    )
    if set(frozen_candidate_hashes) != set(REQUIRED_CANDIDATES):
        blockers.append("candidate_contract_set")
    else:
        for candidate_id, expected in candidate_contract_sha256s.items():
            if (
                str(frozen_candidate_hashes.get(candidate_id) or "").lower()
                != expected.lower()
            ):
                blockers.append(f"{candidate_id}_contract_sha256")

    collection = dict(plan.get("collection") or {})
    if collection.get("market_family") != "btc_updown_5m":
        blockers.append("market_family")
    if int(collection.get("quality_valid_market_target") or 0) != 120:
        blockers.append("quality_valid_market_target")
    if int(collection.get("maximum_attempted_market_count") or 0) != 180:
        blockers.append("maximum_attempted_market_count")
    if int(collection.get("bounded_batch_market_count") or 0) != 12:
        blockers.append("bounded_batch_market_count")
    if (
        int(
            collection.get(
                "strictly_later_minimum_market_start_ts_exclusive"
            )
            or 0
        )
        != freeze_created_ts
    ):
        blockers.append("strictly_later_boundary")
    if collection.get("selection_rule") != (
        "chronological_earliest_quality_valid_after_freeze_boundary"
    ):
        blockers.append("selection_rule")
    for name in (
        "outcomes_resolution_labels_or_pnl_opened",
        "candidate_scoring_during_raw_capture_allowed",
        "settlement_finalizer_enabled_during_collection",
        "resolution_provider_enabled_during_collection",
        "result_dependent_extension_allowed",
    ):
        if collection.get(name) is not False:
            blockers.append(name)

    alpha = dict(plan.get("alpha_spending") or {})
    if int(alpha.get("fresh_attempt_number") or 0) != 1:
        blockers.append("fresh_attempt_number")
    if float(alpha.get("familywise_window_alpha") or 0.0) != 0.025:
        blockers.append("familywise_window_alpha")
    if int(alpha.get("parallel_tested_candidate_count") or 0) != 2:
        blockers.append("parallel_tested_candidate_count")
    if float(alpha.get("per_candidate_alpha") or 0.0) != 0.0125:
        blockers.append("per_candidate_alpha")
    if alpha.get("attempt_consumed") is not False:
        blockers.append("attempt_consumed_before_target_access")

    safety = dict(plan.get("safety") or {})
    if safety.get("paper_only") is not True:
        blockers.append("paper_only")
    for name in (
        "capital_at_risk",
        "polymarket_write_enabled",
        "wallet_signing_enabled",
        "promotion_unlocked",
        "live_unlocked",
    ):
        if safety.get(name) is not False:
            blockers.append(name)

    historical = dict(plan.get("historical_replay_prerequisite") or {})
    historical_completed_ts = int(
        historical.get("evaluation_completed_ts") or 0
    )
    if (
        historical.get("gate_contract_sha256")
        != historical_gate_contract_sha256
    ):
        blockers.append("historical_gate_contract_sha256")
    if historical.get("report_sha256") != historical_replay_report_sha256:
        blockers.append("historical_replay_report_sha256")
    if historical.get("candidate_id") != "v8_1_primary_no_fallback":
        blockers.append("historical_candidate_id")
    if historical.get("historical_superiority_gate_passed") is not True:
        blockers.append("historical_superiority_gate")
    if historical.get("future_collection_prerequisite_satisfied") is not True:
        blockers.append("historical_future_collection_prerequisite")
    if historical.get("historical_replay_is_promotion_evidence") is not False:
        blockers.append("historical_promotion_evidence_role")
    if (
        historical_completed_ts <= 0
        or historical_completed_ts >= freeze_created_ts
    ):
        blockers.append("historical_replay_not_completed_before_freeze")
    if (
        historical_replay_report.get("candidate_id")
        != historical.get("candidate_id")
        or historical_replay_report.get("evaluation_completed_ts")
        != historical_completed_ts
        or historical_replay_report.get("historical_superiority_gate_passed")
        is not True
        or historical_replay_report.get(
            "future_collection_prerequisite_satisfied"
        )
        is not True
        or historical_replay_report.get(
            "historical_replay_is_promotion_evidence"
        )
        is not False
        or not all(
            (historical_replay_report.get("checks") or {}).values()
        )
    ):
        blockers.append("historical_replay_report_content")

    supersession = dict(plan.get("supersession") or {})
    excluded_probe = dict(
        supersession.get("excluded_pre_replay_probe") or {}
    )
    excluded_model_binding_capture = dict(
        supersession.get("excluded_pre_model_binding_capture") or {}
    )
    prior_plan_sha256 = str(
        supersession.get("superseded_collection_plan_sha256") or ""
    )
    if len(prior_plan_sha256) != 64:
        blockers.append("superseded_collection_plan_sha256")
    if excluded_probe.get("excluded_from_fresh_attempt") is not True:
        blockers.append("pre_replay_probe_not_excluded")
    if excluded_probe.get("labels_outcomes_or_pnl_opened") is not False:
        blockers.append("pre_replay_probe_target_access")
    if excluded_probe.get("consumes_attempt_or_alpha") is not False:
        blockers.append("pre_replay_probe_attempt_accounting")
    if int(excluded_probe.get("market_start_ts") or 0) > freeze_created_ts:
        blockers.append("pre_replay_probe_not_before_refreeze")
    if collection.get("service_root") == excluded_probe.get("service_root"):
        blockers.append("pre_replay_probe_service_root_reused")
    if (
        excluded_model_binding_capture.get("excluded_from_fresh_attempt")
        is not True
    ):
        blockers.append("pre_model_binding_capture_not_excluded")
    if (
        excluded_model_binding_capture.get("labels_outcomes_or_pnl_opened")
        is not False
    ):
        blockers.append("pre_model_binding_capture_target_access")
    if (
        excluded_model_binding_capture.get("consumes_attempt_or_alpha")
        is not False
    ):
        blockers.append("pre_model_binding_capture_attempt_accounting")
    if (
        excluded_model_binding_capture.get("index_entry_written")
        is not False
    ):
        blockers.append("pre_model_binding_capture_indexed")
    if (
        int(excluded_model_binding_capture.get("market_start_ts") or 0)
        > freeze_created_ts
    ):
        blockers.append("pre_model_binding_capture_not_before_refreeze")
    if collection.get("service_root") == excluded_model_binding_capture.get(
        "service_root"
    ):
        blockers.append("pre_model_binding_capture_service_root_reused")
    try:
        validate_parallel_frozen_model_binding(
            frozen_model_binding,
            candidate_contracts={},
            expected_binding_sha256=frozen_model_binding_sha256,
        )
    except ParallelFutureGateError:
        blockers.append("frozen_model_binding")
    if blockers:
        raise ParallelFutureGateError(
            "parallel future collection plan invalid: "
            + ", ".join(sorted(blockers))
        )


def validate_parallel_candidate_protocol(
    protocol: dict[str, Any],
    *,
    candidate_contracts: dict[str, dict[str, Any]],
) -> None:
    """Validate frozen candidates, shared window, gates, and multiplicity."""

    blockers: list[str] = []
    if protocol.get("schema_version") != PARALLEL_PROTOCOL_SCHEMA_VERSION:
        blockers.append("schema_version")
    if tuple(protocol.get("candidate_order") or ()) != REQUIRED_CANDIDATES:
        blockers.append("candidate_order")
    if set(candidate_contracts) != set(REQUIRED_CANDIDATES):
        blockers.append("candidate_contract_set")
    if protocol.get("shared_window", {}).get("strictly_later") is not True:
        blockers.append("strictly_later_window")
    if protocol.get("shared_window", {}).get("same_source_rows_for_all_candidates") is not True:
        blockers.append("same_source_rows")
    if protocol.get("multiplicity", {}).get("method") != "bonferroni":
        blockers.append("multiplicity_method")
    if float(protocol.get("multiplicity", {}).get("per_candidate_alpha") or 0.0) != 0.0125:
        blockers.append("per_candidate_alpha")
    if protocol.get("winner_selection_rule") != (
        "highest_adjusted_candidate_minus_baseline_lcb_then_total_pnl_then_candidate_id"
    ):
        blockers.append("winner_selection_rule")
    for candidate_id, contract in candidate_contracts.items():
        if contract.get("candidate_id") != candidate_id:
            blockers.append(f"{candidate_id}_identity")
        if contract.get("contract_frozen_before_target_access") is not True:
            blockers.append(f"{candidate_id}_not_frozen")
        if contract.get("outcomes_labels_settlement_returns_or_pnl_opened") is not False:
            blockers.append(f"{candidate_id}_target_opened")
        if any(contract.get("safety", {}).values()):
            blockers.append(f"{candidate_id}_safety")
    primary_contracts = [
        candidate_contracts.get(candidate_id, {})
        for candidate_id in REQUIRED_CANDIDATES[:2]
    ]
    if any(
        contract.get("source_model_hash_role")
        != "source_training_rows_sha256"
        or not _is_sha256(contract.get("frozen_model_binding_sha256"))
        or not _is_sha256(contract.get("frozen_model_artifact_sha256"))
        or not _is_sha256(contract.get("frozen_model_artifact_id"))
        or not _is_sha256(contract.get("initial_controller_state_id"))
        or float(contract.get("paper_position_size") or 0.0) != 0.2
        for contract in primary_contracts
    ):
        blockers.append("primary_frozen_model_binding")
    if len(
        {
            (
                contract.get("frozen_model_binding_sha256"),
                contract.get("frozen_model_artifact_sha256"),
                contract.get("frozen_model_artifact_id"),
                contract.get("initial_controller_state_id"),
                contract.get("paper_position_size"),
            )
            for contract in primary_contracts
        }
    ) != 1:
        blockers.append("primary_frozen_model_binding_mismatch")
    baseline = candidate_contracts.get("matched_frozen_v6_7", {})
    if (
        not _is_sha256(
            baseline.get("frozen_v6_2_candidate_manifest_sha256")
        )
        or float(baseline.get("paper_position_size") or 0.0) != 0.2
    ):
        blockers.append("baseline_frozen_model_binding")
    if blockers:
        raise ParallelFutureGateError(
            "parallel candidate protocol invalid: " + ", ".join(sorted(blockers))
        )


def validate_parallel_frozen_model_binding(
    binding: dict[str, Any],
    *,
    candidate_contracts: dict[str, dict[str, Any]],
    expected_binding_sha256: str,
) -> None:
    """Validate the immutable v8.1 model and controller-state binding."""

    blockers: list[str] = []
    if binding.get("schema_version") != FROZEN_MODEL_BINDING_SCHEMA_VERSION:
        blockers.append("schema_version")
    if binding.get("candidate_name") != "adaptive_support_controller_v8_1":
        blockers.append("candidate_name")
    if binding.get("primary_candidate_ids") != list(REQUIRED_CANDIDATES[:2]):
        blockers.append("primary_candidate_ids")
    for field in (
        "historical_fit_manifest_sha256",
        "frozen_model_artifact_sha256",
        "frozen_model_artifact_id",
        "frozen_booster_sha256",
        "frozen_profile_sha256",
        "source_training_rows_sha256",
    ):
        if not _is_sha256(binding.get(field)):
            blockers.append(field)
    state = dict(binding.get("initial_controller_state") or {})
    for field in (
        "rank_state_id",
        "rank_lineage_hash",
        "eligible_prediction_scores_hash",
        "controller_guard_acceptance_history_hash",
    ):
        if not _is_sha256(state.get(field)):
            blockers.append(field)
    execution = dict(binding.get("execution") or {})
    if float(execution.get("paper_position_size") or 0.0) != 0.2:
        blockers.append("paper_position_size")
    if execution.get("v8_1_fallback_allowed") is not False:
        blockers.append("v8_1_fallback_allowed")
    if binding.get("frozen_before_fresh_collection") is not True:
        blockers.append("not_frozen_before_collection")
    if binding.get(
        "current_or_future_outcomes_labels_settlement_returns_or_pnl_opened"
    ) is not False:
        blockers.append("target_access")
    if any((binding.get("safety") or {}).values()):
        blockers.append("safety")
    if not _is_sha256(expected_binding_sha256):
        blockers.append("expected_binding_sha256")
    for candidate_id in REQUIRED_CANDIDATES[:2]:
        contract = candidate_contracts.get(candidate_id, {})
        if contract and (
            contract.get("frozen_model_binding_sha256")
            != expected_binding_sha256
            or contract.get("frozen_model_artifact_sha256")
            != binding.get("frozen_model_artifact_sha256")
            or contract.get("frozen_model_artifact_id")
            != binding.get("frozen_model_artifact_id")
            or contract.get("initial_controller_state_id")
            != state.get("rank_state_id")
        ):
            blockers.append(f"{candidate_id}_binding")
    if blockers:
        raise ParallelFutureGateError(
            "parallel frozen model binding invalid: "
            + ", ".join(sorted(set(blockers)))
        )


def build_parallel_target_free_freeze(
    *,
    protocol: dict[str, Any],
    candidate_contracts: dict[str, dict[str, Any]],
    source_rows: list[dict[str, Any]],
    decisions_by_candidate: dict[str, list[dict[str, Any]]],
    decision_freeze_created_ts: int,
    target_access_started: bool,
) -> dict[str, Any]:
    """Freeze all policy decisions on one target-free source-row grid."""

    validate_parallel_candidate_protocol(
        protocol,
        candidate_contracts=candidate_contracts,
    )
    if target_access_started:
        raise ParallelFutureGateError("target access started before parallel decision freeze")
    if decision_freeze_created_ts <= 0:
        raise ParallelFutureGateError("decision_freeze_created_ts must be positive")
    if not source_rows:
        raise ParallelFutureGateError("shared source rows are empty")
    forbidden = sorted(_find_target_fields(source_rows))
    if forbidden:
        raise ParallelFutureGateError(
            "target fields found before decision freeze: " + ", ".join(forbidden)
        )
    source_keys = [_decision_key(row) for row in source_rows]
    if len(source_keys) != len(set(source_keys)):
        raise ParallelFutureGateError("shared source decision grid contains duplicates")
    if set(decisions_by_candidate) != set(REQUIRED_CANDIDATES):
        raise ParallelFutureGateError("parallel decision candidate set is incomplete")
    decision_streams: dict[str, dict[str, Any]] = {}
    for candidate_id in REQUIRED_CANDIDATES:
        rows = list(decisions_by_candidate[candidate_id])
        keys = [_decision_key(row) for row in rows]
        if keys != source_keys:
            raise ParallelFutureGateError(
                f"{candidate_id} does not use the exact shared source-row order"
            )
        for row in rows:
            _validate_decision_row(candidate_id, row)
        stream_hash = canonical_payload_sha256(
            rows,
            payload_schema_version=DECISION_STREAM_SCHEMA_VERSION,
        )
        decision_streams[candidate_id] = {
            "candidate_contract_sha256": canonical_payload_sha256(
                candidate_contracts[candidate_id],
                payload_schema_version=str(
                    candidate_contracts[candidate_id]["schema_version"]
                ),
            ),
            "decision_count": len(rows),
            "decision_stream_sha256": stream_hash,
            "decisions": rows,
        }
    source_rows_sha256 = canonical_payload_sha256(
        source_rows,
        payload_schema_version=str(protocol["shared_window"]["source_row_schema_version"]),
    )
    freeze = {
        "schema_version": PARALLEL_FREEZE_SCHEMA_VERSION,
        "protocol_sha256": canonical_payload_sha256(
            protocol,
            payload_schema_version=PARALLEL_PROTOCOL_SCHEMA_VERSION,
        ),
        "decision_freeze_created_ts": decision_freeze_created_ts,
        "shared_source_row_count": len(source_rows),
        "shared_source_rows_sha256": source_rows_sha256,
        "shared_source_rows": source_rows,
        "candidate_decision_streams": decision_streams,
        "all_candidate_decisions_frozen_before_target_access": True,
        "outcomes_labels_settlement_returns_or_pnl_opened": False,
        "result_selected_extension_threshold_change_candidate_replacement_or_rerun_allowed": False,
        "paper_candidate_unlocked": False,
        "promotion_unlocked": False,
        "live_unlocked": False,
        "write_enabled": False,
        "wallet_enabled": False,
        "capital_at_risk": False,
    }
    freeze["freeze_sha256"] = canonical_payload_sha256(
        freeze,
        payload_schema_version=PARALLEL_FREEZE_SCHEMA_VERSION,
    )
    return freeze


def evaluate_parallel_future_gate(
    *,
    protocol: dict[str, Any],
    freeze: dict[str, Any],
    settled_targets: list[dict[str, Any]],
    evaluation_started_ts: int,
    consumed_freeze_sha256s: set[str] | frozenset[str],
) -> dict[str, Any]:
    """Consume one frozen window once and produce multiplicity-aware reports."""

    _validate_freeze(freeze)
    freeze_sha256 = str(freeze["freeze_sha256"])
    if freeze_sha256 in consumed_freeze_sha256s:
        raise ParallelFutureGateError("parallel future freeze already consumed")
    if evaluation_started_ts <= int(freeze["decision_freeze_created_ts"]):
        raise ParallelFutureGateError("evaluation must start after decision freeze")
    targets = {_decision_key(row): row for row in settled_targets}
    source_keys = [_decision_key(row) for row in freeze["shared_source_rows"]]
    if len(targets) != len(settled_targets) or set(targets) != set(source_keys):
        raise ParallelFutureGateError("settled target grid differs from frozen source grid")
    for target in settled_targets:
        if target.get("target_available_after_decision_freeze") is not True:
            raise ParallelFutureGateError("target was not proven post-freeze")
        if target.get("target_used_as_decision_input") is not False:
            raise ParallelFutureGateError("target contaminated a frozen decision")
    candidate_rows: dict[str, list[dict[str, Any]]] = {}
    metrics: dict[str, dict[str, Any]] = {}
    for candidate_id in REQUIRED_CANDIDATES:
        settled_rows = _settle_candidate_rows(
            candidate_id=candidate_id,
            decisions=freeze["candidate_decision_streams"][candidate_id]["decisions"],
            targets=targets,
        )
        candidate_rows[candidate_id] = settled_rows
        metrics[candidate_id] = _candidate_metrics(settled_rows)
    baseline_by_key = {
        _decision_key(row): float(row["after_cost_pnl"])
        for row in candidate_rows["matched_frozen_v6_7"]
    }
    gates: dict[str, dict[str, Any]] = {}
    candidate_alpha = float(protocol["multiplicity"]["per_candidate_alpha"])
    bootstrap_count = int(protocol["statistical_gate"]["bootstrap_resample_count"])
    bootstrap_seed = int(protocol["statistical_gate"]["bootstrap_seed"])
    minimum_support = int(protocol["statistical_gate"]["minimum_total_support"])
    for offset, candidate_id in enumerate(REQUIRED_CANDIDATES[:2]):
        rows = candidate_rows[candidate_id]
        delta_by_market = defaultdict(float)
        for row in rows:
            key = _decision_key(row)
            delta_by_market[str(row["market_id"])] += (
                float(row["after_cost_pnl"]) - baseline_by_key[key]
            )
        lcb, ucb = _bootstrap_sum_interval(
            dict(delta_by_market),
            alpha=candidate_alpha,
            resample_count=bootstrap_count,
            seed=bootstrap_seed + offset,
        )
        candidate_metrics = metrics[candidate_id]
        delta_total = sum(delta_by_market.values())
        candidate_largest_winner_removed = _largest_winner_removed(
            [float(row["after_cost_pnl"]) for row in rows]
        )
        delta_largest_winner_removed = _largest_winner_removed(
            list(delta_by_market.values())
        )
        support = int(candidate_metrics["accepted_bet_count"])
        status = "evaluated" if support >= minimum_support else "insufficient_support"
        passed = (
            status == "evaluated"
            and float(candidate_metrics["total_after_cost_pnl"]) > 0.0
            and delta_total > 0.0
            and lcb > 0.0
            and candidate_largest_winner_removed > 0.0
            and delta_largest_winner_removed > 0.0
        )
        gates[candidate_id] = {
            "status": status,
            "accepted_bet_count": support,
            "minimum_total_support": minimum_support,
            "total_after_cost_pnl": candidate_metrics["total_after_cost_pnl"],
            "candidate_minus_baseline_after_cost_pnl": delta_total,
            "adjusted_bootstrap_alpha": candidate_alpha,
            "candidate_minus_baseline_bootstrap_lcb": lcb,
            "candidate_minus_baseline_bootstrap_ucb": ucb,
            "candidate_largest_winner_removed_after_cost_pnl": (
                candidate_largest_winner_removed
            ),
            "candidate_minus_baseline_largest_winner_removed_after_cost_pnl": (
                delta_largest_winner_removed
            ),
            "all_hard_gates_passed": passed,
        }
    eligible_candidates = [
        candidate_id
        for candidate_id in REQUIRED_CANDIDATES[:2]
        if gates[candidate_id]["all_hard_gates_passed"] is True
    ]
    selected = (
        sorted(
            eligible_candidates,
            key=lambda candidate_id: (
                -float(gates[candidate_id]["candidate_minus_baseline_bootstrap_lcb"]),
                -float(gates[candidate_id]["total_after_cost_pnl"]),
                candidate_id,
            ),
        )[0]
        if eligible_candidates
        else None
    )
    claim = {
        "schema_version": "bigan-v8-parallel-future-single-use-claim-v1",
        "freeze_sha256": freeze_sha256,
        "evaluation_started_ts": evaluation_started_ts,
        "single_use": True,
        "target_access_after_decision_freeze": True,
        "result_selected_rerun_allowed": False,
    }
    claim["claim_sha256"] = canonical_payload_sha256(
        claim,
        payload_schema_version=str(claim["schema_version"]),
    )
    report = {
        "schema_version": PARALLEL_EVALUATION_SCHEMA_VERSION,
        "single_use_claim": claim,
        "candidate_metrics": metrics,
        "candidate_gates": gates,
        "multiplicity_aware_selected_candidate": selected,
        "selection_rule": protocol["winner_selection_rule"],
        "support_fallback_abstention_attribution": {
            candidate_id: {
                key: value
                for key, value in metrics[candidate_id].items()
                if key
                in {
                    "accepted_bet_count",
                    "primary_count",
                    "fallback_count",
                    "abstention_count",
                    "no_bet_count",
                }
            }
            for candidate_id in REQUIRED_CANDIDATES
        },
        "outcomes_used_for_threshold_candidate_or_extension_selection": False,
        "future_result_driven_rerun_allowed": False,
        "paper_candidate_unlocked": False,
        "promotion_unlocked": False,
        "live_unlocked": False,
        "write_enabled": False,
        "wallet_enabled": False,
        "capital_at_risk": False,
    }
    report["report_sha256"] = canonical_payload_sha256(
        report,
        payload_schema_version=PARALLEL_EVALUATION_SCHEMA_VERSION,
    )
    return {
        "claim": claim,
        "candidate_rows": candidate_rows,
        "report": report,
        "final_manifest": {
            "schema_version": "bigan-v8-parallel-future-final-manifest-v1",
            "freeze_sha256": freeze_sha256,
            "claim_sha256": claim["claim_sha256"],
            "report_sha256": report["report_sha256"],
            "candidate_decision_stream_sha256s": {
                candidate_id: freeze["candidate_decision_streams"][candidate_id][
                    "decision_stream_sha256"
                ]
                for candidate_id in REQUIRED_CANDIDATES
            },
            "candidate_settled_row_sha256s": {
                candidate_id: canonical_payload_sha256(
                    rows,
                    payload_schema_version=PARALLEL_EVALUATION_SCHEMA_VERSION,
                )
                for candidate_id, rows in candidate_rows.items()
            },
            "promotion_unlocked": False,
        },
    }


def _validate_decision_row(candidate_id: str, row: dict[str, Any]) -> None:
    forbidden = _find_target_fields(row)
    if forbidden:
        raise ParallelFutureGateError(
            f"{candidate_id} decision contains target fields: {', '.join(sorted(forbidden))}"
        )
    action = str(row.get("executed_action") or "")
    if not action:
        raise ParallelFutureGateError(f"{candidate_id} decision action missing")
    if row.get("target_used_as_decision_input") is not False:
        raise ParallelFutureGateError(f"{candidate_id} target-free marker missing")
    origin = str(row.get("decision_origin") or "")
    if candidate_id == "v8_1_primary_no_fallback":
        if "fallback" in origin.lower() or row.get("fallback_used") is not False:
            raise ParallelFutureGateError("v8.1 no-fallback candidate used fallback")
        if row.get("primary_abstained") is True and action != "NO_TRADE":
            raise ParallelFutureGateError("v8.1 primary abstention must remain NO_TRADE")
    if (
        candidate_id == "v8_3_primary_with_fallback"
        and row.get("v8_3_frozen_contract_reproduced") is not True
    ):
        raise ParallelFutureGateError("v8.3 frozen behavior reproduction not proven")
    if (
        candidate_id == "matched_frozen_v6_7"
        and row.get("matched_baseline_frozen_contract_reproduced") is not True
    ):
        raise ParallelFutureGateError("matched v6.7 baseline reproduction not proven")


def _validate_freeze(freeze: dict[str, Any]) -> None:
    if freeze.get("schema_version") != PARALLEL_FREEZE_SCHEMA_VERSION:
        raise ParallelFutureGateError("parallel freeze schema invalid")
    expected = canonical_payload_sha256(
        {key: value for key, value in freeze.items() if key != "freeze_sha256"},
        payload_schema_version=PARALLEL_FREEZE_SCHEMA_VERSION,
    )
    if freeze.get("freeze_sha256") != expected:
        raise ParallelFutureGateError("parallel freeze hash mismatch")
    if freeze.get("all_candidate_decisions_frozen_before_target_access") is not True:
        raise ParallelFutureGateError("parallel decisions were not frozen before target access")
    if freeze.get("outcomes_labels_settlement_returns_or_pnl_opened") is not False:
        raise ParallelFutureGateError("parallel freeze is target contaminated")


def _settle_candidate_rows(
    *,
    candidate_id: str,
    decisions: list[dict[str, Any]],
    targets: dict[tuple[str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for decision in decisions:
        target = targets[_decision_key(decision)]
        action = str(decision["executed_action"])
        allowed = bool(decision.get("execution_guard_order_allowed"))
        size = float(decision.get("proposed_order_size") or 0.0)
        target_by_action = dict(target.get("after_cost_pnl_per_notional_by_action") or {})
        if action not in target_by_action:
            raise ParallelFutureGateError("settled target does not cover frozen action")
        pnl = size * float(target_by_action[action]) if allowed and action != "NO_TRADE" else 0.0
        output.append(
            {
                **decision,
                "candidate_id": candidate_id,
                "after_cost_pnl": pnl,
                "target_joined_after_decision_freeze": True,
                "target_used_as_decision_input": False,
            }
        )
    return output


def _candidate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [
        row
        for row in rows
        if row.get("execution_guard_order_allowed") is True
        and row.get("executed_action") != "NO_TRADE"
    ]
    origins = Counter(
        "fallback"
        if "fallback" in str(row.get("decision_origin") or "").lower()
        else "abstention"
        if row.get("executed_action") == "NO_TRADE"
        else "primary"
        for row in rows
    )
    return {
        "decision_count": len(rows),
        "accepted_bet_count": len(accepted),
        "accepted_unique_market_count": len({str(row["market_id"]) for row in accepted}),
        "total_after_cost_pnl": sum(float(row["after_cost_pnl"]) for row in rows),
        "primary_count": origins["primary"],
        "fallback_count": origins["fallback"],
        "abstention_count": origins["abstention"],
        "no_bet_count": len(rows) - len(accepted),
        "side_counts": dict(
            sorted(Counter(str(row.get("selected_side") or "NONE") for row in rows).items())
        ),
        "action_counts": dict(
            sorted(Counter(str(row["executed_action"]) for row in rows).items())
        ),
    }


def _largest_winner_removed(values: list[float]) -> float:
    return sum(values) - max(
        [value for value in values if value > 0.0],
        default=0.0,
    )


def _bootstrap_sum_interval(
    values_by_market: dict[str, float],
    *,
    alpha: float,
    resample_count: int,
    seed: int,
) -> tuple[float, float]:
    if not values_by_market or resample_count <= 0 or not (0.0 < alpha < 0.5):
        raise ParallelFutureGateError("bootstrap contract invalid")
    values = list(values_by_market.values())
    rng = random.Random(seed)
    samples = sorted(
        sum(rng.choice(values) for _ in values) for _ in range(resample_count)
    )
    lower_index = max(0, min(len(samples) - 1, int(alpha * len(samples))))
    upper_index = max(
        0,
        min(len(samples) - 1, int((1.0 - alpha) * len(samples)) - 1),
    )
    return samples[lower_index], samples[upper_index]


def _decision_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["market_id"]), int(row["decision_ts"])


def _is_sha256(value: Any) -> bool:
    candidate = str(value or "").lower()
    return len(candidate) == 64 and all(
        character in "0123456789abcdef" for character in candidate
    )


def _find_target_fields(value: Any, *, path: str = "") -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).lower()
            child_path = f"{path}.{key}" if path else str(key)
            target_like = normalized in TARGET_FIELDS or any(
                token in normalized
                for token in (
                    "resolved_outcome",
                    "settlement_payout",
                    "future_return",
                    "after_cost_pnl",
                    "oracle_action",
                )
            )
            if target_like and not (
                normalized == "target_used_as_decision_input" and item is False
            ):
                found.add(child_path)
            found.update(_find_target_fields(item, path=child_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.update(_find_target_fields(item, path=f"{path}[{index}]"))
    return found


__all__ = [
    "DECISION_STREAM_SCHEMA_VERSION",
    "PARALLEL_COLLECTION_PLAN_SCHEMA_VERSION",
    "PARALLEL_EVALUATION_SCHEMA_VERSION",
    "PARALLEL_FREEZE_SCHEMA_VERSION",
    "PARALLEL_PROTOCOL_SCHEMA_VERSION",
    "ParallelFutureGateError",
    "build_parallel_target_free_freeze",
    "evaluate_parallel_future_gate",
    "validate_parallel_candidate_protocol",
    "validate_parallel_frozen_model_binding",
    "validate_parallel_future_collection_plan",
]
