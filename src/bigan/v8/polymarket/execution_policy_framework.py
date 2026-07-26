"""Bounded, deterministic execution-policy framework for v8 challenge models."""

from __future__ import annotations

import copy
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from bigan.v8.canonical_payload import canonical_payload_sha256

EXECUTION_POLICY_CONTRACT_SCHEMA_VERSION = "bigan-v8-execution-policy-contract-v1"
EXECUTION_POLICY_FIXTURE_SCHEMA_VERSION = "bigan-v8-execution-policy-fixture-v1"
EXECUTION_POLICY_INPUT_SCHEMA_VERSION = "bigan-v8-execution-policy-input-v1"
EXECUTION_POLICY_DECISION_SCHEMA_VERSION = "bigan-v8-execution-policy-decision-v1"
RISK_BUDGET_STATE_SCHEMA_VERSION = "bigan-v8-risk-budget-state-v1"
REPLAY_SCHEMA_VERSION = "bigan-v8-execution-policy-replay-v1"
SOURCE_EXECUTION_COMPATIBILITY_SCHEMA_VERSION = "bigan-v8-source-execution-compatibility-v1"
POLICY_CANDIDATE_MANIFEST_SCHEMA_VERSION = "bigan-v8-policy-candidate-manifest-v1"
EXECUTION_POLICY_ATTRIBUTION_SCHEMA_VERSION = "bigan-v8-execution-policy-attribution-v1"
EXECUTION_POLICY_RECONCILIATION_SCHEMA_VERSION = "bigan-v8-execution-policy-reconciliation-v1"
EXECUTION_POLICY_REPLAY_PARITY_SCHEMA_VERSION = "bigan-v8-execution-policy-replay-parity-v1"
EXECUTION_POLICY_SAFETY_REPORT_SCHEMA_VERSION = "bigan-v8-execution-policy-safety-report-v1"
EXECUTION_POLICY_FUTURE_VALIDATION_SCHEMA_VERSION = (
    "bigan-v8-execution-policy-future-validation-template-v1"
)
BUY_ACTIONS = (
    "BUY_UP_HOLD_TO_SETTLEMENT",
    "BUY_DOWN_HOLD_TO_SETTLEMENT",
    "BUY_UP_SELL_BEFORE_CLOSE",
    "BUY_DOWN_SELL_BEFORE_CLOSE",
)
ALLOWED_ACTIONS = frozenset((*BUY_ACTIONS, "NO_TRADE"))
REQUIRED_DECISION_TIME_INPUTS = frozenset(
    {
        "market_id",
        "decision_ts",
        "source_model_hash",
        "source_action_scores",
        "uncertainty",
        "opportunity_window_id",
        "fill_quality_score",
        "provider_health_score",
        "provider_features_complete",
        "kill_switch_active",
    }
)
REQUIRED_POLICY_CAPABILITIES = (
    "adaptive_abstention",
    "opportunity_budget",
    "global_exposure_cap",
    "side_exposure_cap",
    "market_exposure_cap",
    "duplicate_and_reentry_cooldown",
    "position_replacement",
    "paper_notional_scheduler",
    "fill_quality_filter",
    "provider_health_degradation",
    "separate_policy_kill_switch",
)
REQUIRED_POLICY_SAFETY_FIELDS = frozenset(
    {
        "source_model_mutation_enabled",
        "execution_guard_relaxation_enabled",
        "paper_candidate_unlocked",
        "promotion_unlocked",
        "live_unlocked",
        "write_enabled",
        "wallet_enabled",
        "capital_at_risk",
        "handoff_enabled",
        "source_change_enabled",
        "freeze_change_enabled",
        "#134_resume_allowed",
        "#146_start_allowed",
    }
)
POLICY_CONSTRAINT_FIELDS = frozenset(
    {
        "minimum_source_score",
        "maximum_uncertainty",
        "minimum_fill_quality_score",
        "minimum_provider_health_score",
        "maximum_opportunities_per_window",
        "global_exposure_cap",
        "per_side_exposure_cap",
        "per_market_exposure_cap",
        "reentry_cooldown_ms",
        "replacement_enabled",
        "replacement_minimum_score_uplift",
        "paper_notional_per_position",
        "partial_sizing_allowed",
    }
)
FUTURE_VALIDATION_FREEZE_ORDER = (
    "source_model_hash",
    "execution_policy_hash",
    "compatibility_manifest",
    "target_free_inputs",
    "all_policy_decisions",
    "decision_attribution",
    "risk_budget_state",
    "single_use_target_access_claim",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_INPUT_TOKENS = (
    "outcome",
    "settlement",
    "pnl",
    "profit",
    "future_return",
    "oracle",
    "label",
)


class ExecutionPolicyError(ValueError):
    """Raised when a policy, source input, or state transition is unsafe."""


@dataclass(frozen=True, slots=True)
class _Position:
    market_id: str
    action: str
    side: str
    notional: float
    source_score: float
    opened_ts: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "market_id": self.market_id,
            "action": self.action,
            "side": self.side,
            "notional": self.notional,
            "source_score": self.source_score,
            "opened_ts": self.opened_ts,
        }


def validate_execution_policy_contract(contract: dict[str, Any]) -> None:
    if not isinstance(contract, dict):
        raise ExecutionPolicyError("execution policy contract invalid: contract")
    blockers: list[str] = []
    if set(contract) != {
        "schema_version",
        "issue",
        "interface",
        "source_model_and_execution_policy_independently_versioned",
        "source_scores_mutation_allowed",
        "outcome_driven_policy_search_allowed",
        "unsupported_or_missing_inputs",
        "required_capabilities",
        "replay_parity_required",
        "intent_fill_position_ledger_reconciliation_required",
        "future_comparison_outcome_blind_until_all_decisions_frozen",
        "safety",
    }:
        blockers.append("fields")
    if contract.get("schema_version") != EXECUTION_POLICY_CONTRACT_SCHEMA_VERSION:
        blockers.append("schema_version")
    if contract.get("issue") != 256:
        blockers.append("issue")
    if contract.get("interface") != {
        "input": "frozen_source_action_scores_plus_causal_risk_state",
        "output": "explicit_action_or_no_trade_with_attribution",
        "state_transition": "deterministic_paper_only_risk_budget_state",
        "source_model_hash_required": True,
        "execution_policy_hash_required": True,
    }:
        blockers.append("interface")
    if contract.get("source_model_and_execution_policy_independently_versioned") is not True:
        blockers.append("independent_versioning")
    if contract.get("source_scores_mutation_allowed") is not False:
        blockers.append("source_score_mutation")
    if contract.get("outcome_driven_policy_search_allowed") is not False:
        blockers.append("outcome_search")
    if contract.get("unsupported_or_missing_inputs") != "fail_closed_no_trade":
        blockers.append("missing_input_behavior")
    capabilities = contract.get("required_capabilities")
    if not isinstance(capabilities, list) or tuple(capabilities) != REQUIRED_POLICY_CAPABILITIES:
        blockers.append("required_capabilities")
    if contract.get("replay_parity_required") is not True:
        blockers.append("replay_parity")
    if contract.get("intent_fill_position_ledger_reconciliation_required") is not True:
        blockers.append("reconciliation")
    if contract.get("future_comparison_outcome_blind_until_all_decisions_frozen") is not True:
        blockers.append("future_outcome_blindness")
    safety = contract.get("safety")
    if (
        not isinstance(safety, dict)
        or set(safety) != REQUIRED_POLICY_SAFETY_FIELDS
        or any(value is not False for value in safety.values())
    ):
        blockers.append("safety")
    if blockers:
        raise ExecutionPolicyError(
            "execution policy contract invalid: " + ", ".join(sorted(blockers))
        )


def validate_execution_policy_fixture(policy: dict[str, Any]) -> None:
    """Validate all safety and budget semantics before hashing or execution."""

    if not isinstance(policy, dict):
        raise ExecutionPolicyError("execution policy fixture invalid: policy")
    blockers: list[str] = []
    if set(policy) != {
        "schema_version",
        "candidate_id",
        "description",
        "constraints",
        "source_scores_read_only",
        "paper_only",
        "capital_at_risk",
    }:
        blockers.append("fields")
    if policy.get("schema_version") != EXECUTION_POLICY_FIXTURE_SCHEMA_VERSION:
        blockers.append("schema_version")
    if not isinstance(policy.get("candidate_id"), str) or not policy["candidate_id"].strip():
        blockers.append("candidate_id")
    if not isinstance(policy.get("description"), str) or not policy["description"].strip():
        blockers.append("description")
    if policy.get("source_scores_read_only") is not True:
        blockers.append("source_scores_read_only")
    if policy.get("paper_only") is not True:
        blockers.append("paper_only")
    if policy.get("capital_at_risk") is not False:
        blockers.append("capital_at_risk")
    constraints = policy.get("constraints")
    if not isinstance(constraints, dict) or set(constraints) != POLICY_CONSTRAINT_FIELDS:
        blockers.append("constraints")
        constraints = {}
    unit_interval_fields = (
        "minimum_source_score",
        "maximum_uncertainty",
        "minimum_fill_quality_score",
        "minimum_provider_health_score",
    )
    for field in unit_interval_fields:
        value = constraints.get(field)
        if not _finite_number(value) or not 0.0 <= float(value) <= 1.0:
            blockers.append(field)
    for field in (
        "global_exposure_cap",
        "per_side_exposure_cap",
        "per_market_exposure_cap",
        "paper_notional_per_position",
    ):
        value = constraints.get(field)
        if not _finite_number(value) or float(value) <= 0.0:
            blockers.append(field)
    uplift = constraints.get("replacement_minimum_score_uplift")
    if not _finite_number(uplift) or float(uplift) < 0.0:
        blockers.append("replacement_minimum_score_uplift")
    maximum_opportunities = constraints.get("maximum_opportunities_per_window")
    if (
        isinstance(maximum_opportunities, bool)
        or not isinstance(maximum_opportunities, int)
        or maximum_opportunities <= 0
    ):
        blockers.append("maximum_opportunities_per_window")
    cooldown = constraints.get("reentry_cooldown_ms")
    if isinstance(cooldown, bool) or not isinstance(cooldown, int) or cooldown < 0:
        blockers.append("reentry_cooldown_ms")
    for field in ("replacement_enabled", "partial_sizing_allowed"):
        if not isinstance(constraints.get(field), bool):
            blockers.append(field)
    if not blockers:
        desired = float(constraints["paper_notional_per_position"])
        global_cap = float(constraints["global_exposure_cap"])
        side_cap = float(constraints["per_side_exposure_cap"])
        market_cap = float(constraints["per_market_exposure_cap"])
        if side_cap > global_cap:
            blockers.append("per_side_exposure_cap")
        if market_cap > side_cap or market_cap > global_cap:
            blockers.append("per_market_exposure_cap")
        if constraints["partial_sizing_allowed"] is False and desired > min(
            global_cap, side_cap, market_cap
        ):
            blockers.append("paper_notional_per_position")
    if blockers:
        raise ExecutionPolicyError(
            "execution policy fixture invalid: " + ", ".join(sorted(set(blockers)))
        )


def execution_policy_hash(policy: dict[str, Any]) -> str:
    validate_execution_policy_fixture(policy)
    return canonical_payload_sha256(
        policy,
        payload_schema_version=EXECUTION_POLICY_FIXTURE_SCHEMA_VERSION,
    )


def validate_source_execution_compatibility(
    *,
    compatibility_manifest: dict[str, Any],
    policy: dict[str, Any],
    source_model_hash: str,
) -> None:
    if not isinstance(compatibility_manifest, dict):
        raise ExecutionPolicyError("source/execution compatibility invalid: compatibility_manifest")
    blockers: list[str] = []
    if set(compatibility_manifest) != {
        "schema_version",
        "source_model_id",
        "source_model_hash",
        "source_action_scores_are_read_only",
        "target_or_outcome_inputs_allowed",
        "required_decision_time_inputs",
        "canonical_payload_contract_sha256",
        "feature_missingness_contract_sha256",
        "allowed_execution_policy_hashes",
    }:
        blockers.append("fields")
    if (
        compatibility_manifest.get("schema_version")
        != SOURCE_EXECUTION_COMPATIBILITY_SCHEMA_VERSION
    ):
        blockers.append("schema_version")
    if not _sha256(source_model_hash):
        blockers.append("source_model_hash_format")
    if compatibility_manifest.get("source_model_hash") != source_model_hash:
        blockers.append("source_model_hash")
    if (
        not isinstance(compatibility_manifest.get("source_model_id"), str)
        or not str(compatibility_manifest.get("source_model_id") or "").strip()
    ):
        blockers.append("source_model_id")
    policy_hash = execution_policy_hash(policy)
    allowed_hashes = compatibility_manifest.get("allowed_execution_policy_hashes")
    if (
        not isinstance(allowed_hashes, list)
        or len(allowed_hashes) != 3
        or any(not _sha256(value) for value in allowed_hashes)
        or len(set(allowed_hashes)) != 3
    ):
        blockers.append("allowed_execution_policy_hashes")
        allowed_hashes = []
    if policy_hash not in set(allowed_hashes):
        blockers.append("execution_policy_hash")
    required_inputs = compatibility_manifest.get("required_decision_time_inputs")
    required = (
        set(required_inputs)
        if isinstance(required_inputs, list)
        and all(isinstance(value, str) for value in required_inputs)
        else set()
    )
    if required != REQUIRED_DECISION_TIME_INPUTS:
        blockers.append("required_inputs")
    if compatibility_manifest.get("source_action_scores_are_read_only") is not True:
        blockers.append("source_scores_read_only")
    if compatibility_manifest.get("target_or_outcome_inputs_allowed") is not False:
        blockers.append("target_inputs")
    for field in (
        "canonical_payload_contract_sha256",
        "feature_missingness_contract_sha256",
    ):
        if not _sha256(compatibility_manifest.get(field)):
            blockers.append(field)
    if blockers:
        raise ExecutionPolicyError(
            "source/execution compatibility invalid: " + ", ".join(sorted(blockers))
        )


def validate_policy_candidate_manifest(manifest: dict[str, Any]) -> None:
    """Validate the bounded three-fixture family without reading outcomes."""

    if not isinstance(manifest, dict):
        raise ExecutionPolicyError("policy candidate manifest invalid: manifest")
    blockers: list[str] = []
    if set(manifest) != {
        "schema_version",
        "issue",
        "source_model_hash",
        "candidate_fixtures",
        "candidate_count_cap",
        "open_ended_optimizer_enabled",
        "outcome_selected_candidate_enabled",
        "framework_creation_unlocks_candidate",
        "promotion_unlocked",
    }:
        blockers.append("fields")
    if manifest.get("schema_version") != POLICY_CANDIDATE_MANIFEST_SCHEMA_VERSION:
        blockers.append("schema_version")
    if manifest.get("issue") != 256:
        blockers.append("issue")
    if not _sha256(manifest.get("source_model_hash")):
        blockers.append("source_model_hash")
    fixtures = manifest.get("candidate_fixtures")
    if not isinstance(fixtures, list) or len(fixtures) != 3:
        blockers.append("candidate_fixtures")
        fixtures = []
    if manifest.get("candidate_count_cap") != 3:
        blockers.append("candidate_count_cap")
    for field in (
        "open_ended_optimizer_enabled",
        "outcome_selected_candidate_enabled",
        "framework_creation_unlocks_candidate",
        "promotion_unlocked",
    ):
        if manifest.get(field) is not False:
            blockers.append(field)
    candidate_ids: list[str] = []
    policy_hashes: list[str] = []
    paths: list[str] = []
    for descriptor in fixtures:
        if not isinstance(descriptor, dict) or set(descriptor) != {
            "candidate_id",
            "path",
            "raw_sha256",
            "execution_policy_hash",
        }:
            blockers.append("candidate_descriptor")
            continue
        candidate_id = descriptor.get("candidate_id")
        path = descriptor.get("path")
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            blockers.append("candidate_id")
        else:
            candidate_ids.append(candidate_id)
        if (
            not isinstance(path, str)
            or not path.endswith(".json")
            or "/" in path
            or "\\" in path
            or path in {".json", "..json"}
        ):
            blockers.append("candidate_path")
        else:
            paths.append(path)
        if not _sha256(descriptor.get("raw_sha256")):
            blockers.append("candidate_raw_sha256")
        if not _sha256(descriptor.get("execution_policy_hash")):
            blockers.append("candidate_execution_policy_hash")
        else:
            policy_hashes.append(str(descriptor["execution_policy_hash"]))
    if len(set(candidate_ids)) != len(fixtures):
        blockers.append("candidate_ids_unique")
    if len(set(policy_hashes)) != len(fixtures):
        blockers.append("candidate_policy_hashes_unique")
    if len(set(paths)) != len(fixtures):
        blockers.append("candidate_paths_unique")
    if blockers:
        raise ExecutionPolicyError(
            "policy candidate manifest invalid: " + ", ".join(sorted(set(blockers)))
        )


def validate_execution_policy_future_validation_protocol(
    protocol: dict[str, Any],
) -> None:
    """Validate that the future gate remains preregistered and outcome blind."""

    if not isinstance(protocol, dict):
        raise ExecutionPolicyError("execution policy future validation protocol invalid: protocol")
    expected_fields = {
        "schema_version",
        "candidate_family_manifest",
        "candidate_budget_protocol",
        "family_error_control_contract",
        "parallel_candidate_protocol",
        "regime_definition_contract",
        "freeze_order",
        "outcome_access_before_all_decisions_frozen",
        "result_driven_policy_search_allowed",
        "promotion_unlocked",
    }
    expected_references = {
        "candidate_family_manifest": "candidate_family_manifest.json",
        "candidate_budget_protocol": "candidate_budget_protocol.json",
        "family_error_control_contract": "family_error_control_contract.json",
        "parallel_candidate_protocol": "parallel_candidate_protocol.json",
        "regime_definition_contract": "regime_definition_contract.json",
    }
    freeze_order = protocol.get("freeze_order")
    checks = {
        "fields": set(protocol) == expected_fields,
        "schema_version": protocol.get("schema_version")
        == EXECUTION_POLICY_FUTURE_VALIDATION_SCHEMA_VERSION,
        "references": all(protocol.get(key) == value for key, value in expected_references.items()),
        "freeze_order": isinstance(freeze_order, list)
        and tuple(freeze_order) == FUTURE_VALIDATION_FREEZE_ORDER,
        "outcome_access": protocol.get("outcome_access_before_all_decisions_frozen") is False,
        "result_search": protocol.get("result_driven_policy_search_allowed") is False,
        "promotion": protocol.get("promotion_unlocked") is False,
    }
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ExecutionPolicyError(
            "execution policy future validation protocol invalid: " + ", ".join(blockers)
        )


def run_execution_policy_replay(
    *,
    inputs: list[dict[str, Any]],
    policy: dict[str, Any],
    compatibility_manifest: dict[str, Any],
    runtime_mode: str,
) -> dict[str, Any]:
    """Run the same state machine for offline replay or paper runtime."""

    if runtime_mode not in {"offline_replay", "paper_runtime"}:
        raise ExecutionPolicyError("unsupported runtime_mode")
    if not isinstance(inputs, list):
        raise ExecutionPolicyError("execution policy inputs must be a list")
    policy_hash = execution_policy_hash(policy)
    source_model_hash = str(compatibility_manifest.get("source_model_hash") or "")
    validate_source_execution_compatibility(
        compatibility_manifest=compatibility_manifest,
        policy=policy,
        source_model_hash=source_model_hash,
    )
    state: dict[str, Any] = {
        "positions": {},
        "accepted_by_window": Counter(),
        "last_action_ts_by_market": {},
    }
    decisions: list[dict[str, Any]] = []
    attributions: list[dict[str, Any]] = []
    risk_states: list[dict[str, Any]] = []
    intents: list[dict[str, Any]] = []
    fills: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    previous_decision_ts = -1
    source_input_hashes: list[str] = []
    previous_risk_state_sha256 = "0" * 64
    for sequence, raw_source_input in enumerate(inputs, start=1):
        source_input = copy.deepcopy(raw_source_input) if isinstance(raw_source_input, dict) else {}
        source_input_sha256 = _safe_canonical_hash(
            source_input,
            payload_schema_version=EXECUTION_POLICY_INPUT_SCHEMA_VERSION,
        )
        source_input_hashes.append(source_input_sha256)
        decision_ts_is_valid = _valid_decision_timestamp(source_input.get("decision_ts"))
        decision_ts = _decision_timestamp(source_input.get("decision_ts"))
        if (
            decision_ts_is_valid
            and previous_decision_ts >= 0
            and decision_ts < previous_decision_ts
        ):
            raise ExecutionPolicyError("policy replay inputs must be chronological")
        if decision_ts_is_valid:
            previous_decision_ts = decision_ts
        before = _state_snapshot(state)
        source_scores_before = copy.deepcopy(source_input.get("source_action_scores"))
        decision, attribution, transitions = _decide(
            source_input=source_input,
            policy=policy,
            policy_hash=policy_hash,
            source_model_hash=source_model_hash,
            state=state,
            sequence=sequence,
            source_input_sha256=source_input_sha256,
        )
        if not _structural_equal(
            source_scores_before,
            source_input.get("source_action_scores"),
        ):
            raise ExecutionPolicyError("execution policy mutated source action scores")
        decisions.append(decision)
        attributions.append(attribution)
        for transition in transitions:
            intent_id = f"{policy['candidate_id']}:{sequence}:{transition['transition_type']}"
            intent = {
                "intent_id": intent_id,
                "market_id": transition["market_id"],
                "side": transition["side"],
                "notional_delta": transition["notional_delta"],
                "sequence": sequence,
                "decision_sha256": decision["decision_sha256"],
                "transition_type": transition["transition_type"],
                "paper_only": True,
                "capital_at_risk": False,
            }
            fill = {
                "fill_id": f"fill:{intent_id}",
                "intent_id": intent_id,
                "market_id": transition["market_id"],
                "side": transition["side"],
                "notional_delta": transition["notional_delta"],
                "sequence": sequence,
                "decision_sha256": decision["decision_sha256"],
                "transition_type": transition["transition_type"],
                "paper_only": True,
                "capital_at_risk": False,
            }
            entry = {
                "ledger_entry_id": f"ledger:{intent_id}",
                "fill_id": fill["fill_id"],
                "market_id": transition["market_id"],
                "side": transition["side"],
                "notional_delta": transition["notional_delta"],
                "sequence": sequence,
                "decision_sha256": decision["decision_sha256"],
                "transition_type": transition["transition_type"],
                "paper_only": True,
                "capital_at_risk": False,
            }
            intents.append(intent)
            fills.append(fill)
            ledger.append(entry)
        after = _state_snapshot(state)
        risk_row = {
            "schema_version": RISK_BUDGET_STATE_SCHEMA_VERSION,
            "sequence": sequence,
            "market_id": str(source_input.get("market_id") or ""),
            "decision_ts": decision_ts,
            "execution_policy_hash": policy_hash,
            "decision_sha256": decision["decision_sha256"],
            "source_input_sha256": source_input_sha256,
            "previous_risk_budget_state_sha256": previous_risk_state_sha256,
            "before": before,
            "after": after,
        }
        risk_row["risk_budget_state_sha256"] = canonical_payload_sha256(
            risk_row,
            payload_schema_version=RISK_BUDGET_STATE_SCHEMA_VERSION,
        )
        previous_risk_state_sha256 = risk_row["risk_budget_state_sha256"]
        risk_states.append(risk_row)
    replay = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "runtime_mode": runtime_mode,
        "source_model_hash": source_model_hash,
        "execution_policy_hash": policy_hash,
        "candidate_id": policy["candidate_id"],
        "decisions": decisions,
        "decision_attribution": attributions,
        "risk_budget_state": risk_states,
        "intents": intents,
        "fills": fills,
        "positions": [position.to_dict() for _, position in sorted(state["positions"].items())],
        "ledger": ledger,
        "source_input_stream_sha256": canonical_payload_sha256(
            source_input_hashes,
            payload_schema_version=EXECUTION_POLICY_INPUT_SCHEMA_VERSION,
        ),
        "source_scores_mutated": False,
        "outcomes_settlement_pnl_future_returns_or_oracle_actions_used": False,
        "paper_only": True,
        "paper_candidate_unlocked": False,
        "promotion_unlocked": False,
        "live_unlocked": False,
        "write_enabled": False,
        "wallet_enabled": False,
        "capital_at_risk": False,
    }
    replay["reconciliation_report"] = validate_policy_reconciliation(replay)
    replay["decision_stream_sha256"] = canonical_payload_sha256(
        decisions,
        payload_schema_version=EXECUTION_POLICY_DECISION_SCHEMA_VERSION,
    )
    replay["risk_state_stream_sha256"] = canonical_payload_sha256(
        risk_states,
        payload_schema_version=RISK_BUDGET_STATE_SCHEMA_VERSION,
    )
    replay["attribution_stream_sha256"] = canonical_payload_sha256(
        attributions,
        payload_schema_version=EXECUTION_POLICY_ATTRIBUTION_SCHEMA_VERSION,
    )
    replay["execution_output_sha256"] = _execution_output_sha256(replay)
    return replay


def build_replay_parity_report(
    *,
    offline_replay: dict[str, Any],
    paper_runtime: dict[str, Any],
) -> dict[str, Any]:
    """Require exact policy decision and state parity across both surfaces."""

    validate_execution_policy_replay(offline_replay)
    validate_execution_policy_replay(paper_runtime)
    checks = {
        "decisions_present": bool(offline_replay.get("decisions"))
        and bool(paper_runtime.get("decisions")),
        "offline_runtime_mode_exact": offline_replay.get("runtime_mode") == "offline_replay",
        "paper_runtime_mode_exact": paper_runtime.get("runtime_mode") == "paper_runtime",
        "source_model_hash_match": offline_replay.get("source_model_hash")
        == paper_runtime.get("source_model_hash"),
        "execution_policy_hash_match": offline_replay.get("execution_policy_hash")
        == paper_runtime.get("execution_policy_hash"),
        "candidate_id_match": offline_replay.get("candidate_id")
        == paper_runtime.get("candidate_id"),
        "source_input_stream_sha256_match": _derived_source_input_stream_sha256(offline_replay)
        == _derived_source_input_stream_sha256(paper_runtime),
        "decision_stream_sha256_match": _derived_decision_stream_sha256(offline_replay)
        == _derived_decision_stream_sha256(paper_runtime),
        "attribution_stream_sha256_match": _derived_attribution_stream_sha256(offline_replay)
        == _derived_attribution_stream_sha256(paper_runtime),
        "risk_state_stream_sha256_match": _derived_risk_state_stream_sha256(offline_replay)
        == _derived_risk_state_stream_sha256(paper_runtime),
        "execution_output_sha256_match": _execution_output_sha256(offline_replay)
        == _execution_output_sha256(paper_runtime),
        "offline_reconciliation_passed": offline_replay.get("reconciliation_report", {}).get(
            "passed"
        )
        is True,
        "paper_reconciliation_passed": paper_runtime.get("reconciliation_report", {}).get("passed")
        is True,
    }
    return {
        "schema_version": EXECUTION_POLICY_REPLAY_PARITY_SCHEMA_VERSION,
        "checks": checks,
        "passed": all(checks.values()),
        "offline_runtime_mode": offline_replay.get("runtime_mode"),
        "paper_runtime_mode": paper_runtime.get("runtime_mode"),
        "paper_candidate_unlocked": False,
        "promotion_unlocked": False,
        "live_unlocked": False,
        "capital_at_risk": False,
    }


def build_policy_safety_report(replay: dict[str, Any]) -> dict[str, Any]:
    validate_execution_policy_replay(replay)
    decisions = list(replay.get("decisions") or [])
    attributions = list(replay.get("decision_attribution") or [])
    paper_rows = [
        *decisions,
        *list(replay.get("intents") or []),
        *list(replay.get("fills") or []),
        *list(replay.get("ledger") or []),
    ]
    reason_counts = Counter(
        reason for decision in decisions for reason in decision.get("reason_codes", [])
    )
    kill_switch_is_separate = all(
        attribution.get("kill_switch_active") is not True
        or (
            decision.get("selected_action") == "NO_TRADE"
            and "policy_kill_switch_active_or_missing" in set(decision.get("reason_codes") or [])
            and decision.get("source_action_scores_sha256")
            == attribution.get("source_action_scores_sha256")
        )
        for decision, attribution in zip(decisions, attributions, strict=True)
    )
    checks = {
        "decisions_present": bool(decisions),
        "source_scores_immutable": replay.get("source_scores_mutated") is False
        and all(decision.get("source_scores_mutated") is False for decision in decisions)
        and all(attribution.get("source_scores_mutated") is False for attribution in attributions),
        "no_target_or_outcome_inputs": replay.get(
            "outcomes_settlement_pnl_future_returns_or_oracle_actions_used"
        )
        is False
        and all(decision.get("target_used_as_decision_input") is False for decision in decisions),
        "paper_only": replay.get("paper_only") is True
        and all(row.get("paper_only") is True for row in paper_rows),
        "capital_at_risk_false": replay.get("capital_at_risk") is False
        and all(row.get("capital_at_risk") is False for row in paper_rows),
        "reconciliation_passed": replay.get("reconciliation_report", {}).get("passed") is True,
        "all_no_trade_has_reason": all(
            decision.get("selected_action") != "NO_TRADE" or bool(decision.get("reason_codes"))
            for decision in decisions
        ),
        "policy_kill_switch_separate_from_source_scores": kill_switch_is_separate,
    }
    return {
        "schema_version": EXECUTION_POLICY_SAFETY_REPORT_SCHEMA_VERSION,
        "checks": checks,
        "passed": all(checks.values()),
        "reason_counts": dict(sorted(reason_counts.items())),
        "policy_kill_switch_separate_from_source_scores": kill_switch_is_separate,
        "paper_candidate_unlocked": False,
        "promotion_unlocked": False,
        "live_unlocked": False,
        "write_enabled": False,
        "wallet_enabled": False,
        "capital_at_risk": False,
    }


def validate_policy_reconciliation(replay: dict[str, Any]) -> dict[str, Any]:
    intents = list(replay.get("intents") or [])
    fills = list(replay.get("fills") or [])
    ledger = list(replay.get("ledger") or [])
    positions = list(replay.get("positions") or [])
    decisions = list(replay.get("decisions") or [])
    risk_states = list(replay.get("risk_budget_state") or [])
    intent_by_id = {str(row.get("intent_id")): row for row in intents}
    fill_by_id = {str(row.get("fill_id")): row for row in fills}
    ledger_by_id = {str(row.get("ledger_entry_id")): row for row in ledger}
    position_by_market = {str(row.get("market_id")): row for row in positions}
    transition_rows_valid = (
        all(_valid_transition_row(row, id_field="intent_id") for row in intents)
        and all(_valid_transition_row(row, id_field="fill_id") for row in fills)
        and all(_valid_transition_row(row, id_field="ledger_entry_id") for row in ledger)
    )
    intent_fill_match = (
        len(fills) == len(intents)
        and {str(row.get("intent_id")) for row in fills} == set(intent_by_id)
        and all(
            _transition_rows_match(
                row,
                intent_by_id.get(str(row.get("intent_id"))),
            )
            for row in fills
        )
    )
    fill_ledger_match = (
        len(ledger) == len(fills)
        and {str(row.get("fill_id")) for row in ledger} == set(fill_by_id)
        and all(
            _transition_rows_match(
                row,
                fill_by_id.get(str(row.get("fill_id"))),
            )
            for row in ledger
        )
    )
    identifier_chain_valid = (
        all(
            row.get("intent_id")
            == f"{replay.get('candidate_id')}:{row.get('sequence')}:{row.get('transition_type')}"
            for row in intents
        )
        and all(row.get("fill_id") == f"fill:{row.get('intent_id')}" for row in fills)
        and all(
            row.get("ledger_entry_id")
            == f"ledger:{str(row.get('fill_id') or '').removeprefix('fill:')}"
            for row in ledger
        )
    )
    ledger_market = _sum_notional(ledger, "market_id")
    ledger_side = _sum_notional(ledger, "side")
    position_market = _sum_notional(positions, "market_id", value_field="notional")
    position_side = _sum_notional(positions, "side", value_field="notional")
    per_decision_transitions = _decision_transition_checks(
        decisions=decisions,
        ledger=ledger,
        candidate_id=str(replay.get("candidate_id") or ""),
    )
    state_transition_reconciliation = _risk_state_transition_checks(
        risk_states=risk_states,
        ledger=ledger,
    )
    checks = {
        "transition_rows_valid": transition_rows_valid,
        "intent_ids_unique": len(intent_by_id) == len(intents) and "" not in intent_by_id,
        "fill_ids_unique": len(fill_by_id) == len(fills) and "" not in fill_by_id,
        "ledger_entry_ids_unique": len(ledger_by_id) == len(ledger) and "" not in ledger_by_id,
        "position_market_ids_unique": len(position_by_market) == len(positions)
        and "" not in position_by_market,
        "one_fill_per_intent": intent_fill_match,
        "one_ledger_entry_per_fill": fill_ledger_match,
        "identifier_chain_valid": identifier_chain_valid,
        "intent_fill_fields_match": intent_fill_match,
        "fill_ledger_fields_match": fill_ledger_match,
        "decision_transition_shape_valid": per_decision_transitions,
        "risk_state_transitions_reconcile": state_transition_reconciliation,
        "ledger_market_exposure_reconciles_positions": _numeric_maps_equal(
            ledger_market, position_market
        ),
        "ledger_side_exposure_reconciles_positions": _numeric_maps_equal(
            ledger_side, position_side
        ),
        "ledger_total_exposure_reconciles_positions": _close(
            sum(ledger_market.values()), sum(position_market.values())
        ),
        "positions_valid": all(_valid_position(row) for row in positions),
        "paper_only": all(row.get("paper_only") is True for row in (*intents, *fills, *ledger)),
        "capital_at_risk_false": all(
            row.get("capital_at_risk") is False for row in (*intents, *fills, *ledger)
        ),
    }
    report = {
        "schema_version": EXECUTION_POLICY_RECONCILIATION_SCHEMA_VERSION,
        "checks": checks,
        "passed": all(checks.values()),
        "intent_count": len(intents),
        "fill_count": len(fills),
        "position_count": len(positions),
        "ledger_entry_count": len(ledger),
        "paper_candidate_unlocked": False,
        "promotion_unlocked": False,
        "live_unlocked": False,
        "write_enabled": False,
        "wallet_enabled": False,
        "capital_at_risk": False,
    }
    if not report["passed"]:
        raise ExecutionPolicyError("policy position/intent/fill/ledger reconciliation failed")
    return report


def validate_execution_policy_replay(replay: dict[str, Any]) -> None:
    """Recompute all row, stream, state-chain, and ledger evidence."""

    blockers: list[str] = []
    if not isinstance(replay, dict):
        raise ExecutionPolicyError("execution policy replay invalid: replay")
    if replay.get("schema_version") != REPLAY_SCHEMA_VERSION:
        blockers.append("schema_version")
    if set(replay) != {
        "schema_version",
        "runtime_mode",
        "source_model_hash",
        "execution_policy_hash",
        "candidate_id",
        "decisions",
        "decision_attribution",
        "risk_budget_state",
        "intents",
        "fills",
        "positions",
        "ledger",
        "source_input_stream_sha256",
        "source_scores_mutated",
        "outcomes_settlement_pnl_future_returns_or_oracle_actions_used",
        "paper_only",
        "paper_candidate_unlocked",
        "promotion_unlocked",
        "live_unlocked",
        "write_enabled",
        "wallet_enabled",
        "capital_at_risk",
        "reconciliation_report",
        "decision_stream_sha256",
        "risk_state_stream_sha256",
        "attribution_stream_sha256",
        "execution_output_sha256",
    }:
        blockers.append("fields")
    if replay.get("runtime_mode") not in {"offline_replay", "paper_runtime"}:
        blockers.append("runtime_mode")
    source_model_hash = replay.get("source_model_hash")
    execution_policy_sha256 = replay.get("execution_policy_hash")
    candidate_id = replay.get("candidate_id")
    if not _sha256(source_model_hash):
        blockers.append("source_model_hash")
    if not _sha256(execution_policy_sha256):
        blockers.append("execution_policy_hash")
    if not isinstance(candidate_id, str) or not candidate_id:
        blockers.append("candidate_id")
    row_fields = (
        "decisions",
        "decision_attribution",
        "risk_budget_state",
        "intents",
        "fills",
        "positions",
        "ledger",
    )
    if any(not isinstance(replay.get(field), list) for field in row_fields):
        blockers.append("row_collections")
    decisions = replay.get("decisions") if isinstance(replay.get("decisions"), list) else []
    attributions = (
        replay.get("decision_attribution")
        if isinstance(replay.get("decision_attribution"), list)
        else []
    )
    risk_states = (
        replay.get("risk_budget_state") if isinstance(replay.get("risk_budget_state"), list) else []
    )
    if not (len(decisions) == len(attributions) == len(risk_states)):
        blockers.append("decision_attribution_risk_counts")
    previous_after = _empty_state_snapshot()
    previous_risk_sha256 = "0" * 64
    for sequence, rows in enumerate(
        zip(decisions, attributions, risk_states, strict=False), start=1
    ):
        decision, attribution, risk_row = rows
        if not _valid_decision(
            decision,
            sequence=sequence,
            candidate_id=str(candidate_id or ""),
            source_model_hash=str(source_model_hash or ""),
            execution_policy_sha256=str(execution_policy_sha256 or ""),
        ):
            blockers.append(f"decision_{sequence}")
        if not _valid_attribution(attribution, decision=decision, sequence=sequence):
            blockers.append(f"attribution_{sequence}")
        if not _valid_risk_state_row(
            risk_row,
            decision=decision,
            sequence=sequence,
            execution_policy_sha256=str(execution_policy_sha256 or ""),
            previous_after=previous_after,
            previous_risk_sha256=previous_risk_sha256,
        ):
            blockers.append(f"risk_state_{sequence}")
        if isinstance(risk_row, dict):
            after = risk_row.get("after")
            if isinstance(after, dict):
                previous_after = after
            previous_risk_sha256 = str(risk_row.get("risk_budget_state_sha256") or "")
    final_positions = replay.get("positions") if isinstance(replay.get("positions"), list) else []
    if final_positions != previous_after.get("positions"):
        blockers.append("final_positions")
    if replay.get("source_scores_mutated") is not False:
        blockers.append("source_scores_mutated")
    if replay.get("outcomes_settlement_pnl_future_returns_or_oracle_actions_used") is not False:
        blockers.append("target_or_future_inputs")
    if replay.get("paper_only") is not True:
        blockers.append("paper_only")
    for field in (
        "paper_candidate_unlocked",
        "promotion_unlocked",
        "live_unlocked",
        "write_enabled",
        "wallet_enabled",
        "capital_at_risk",
    ):
        if replay.get(field) is not False:
            blockers.append(field)
    try:
        digest_checks = {
            "source_input_stream_sha256": _derived_source_input_stream_sha256(replay),
            "decision_stream_sha256": _derived_decision_stream_sha256(replay),
            "attribution_stream_sha256": _derived_attribution_stream_sha256(replay),
            "risk_state_stream_sha256": _derived_risk_state_stream_sha256(replay),
            "execution_output_sha256": _execution_output_sha256(replay),
        }
    except (TypeError, ValueError, OverflowError):
        blockers.append("stream_encoding")
    else:
        for field, expected in digest_checks.items():
            if replay.get(field) != expected:
                blockers.append(field)
    try:
        derived_reconciliation = validate_policy_reconciliation(replay)
    except (ExecutionPolicyError, KeyError, TypeError, ValueError, OverflowError):
        blockers.append("reconciliation")
    else:
        if replay.get("reconciliation_report") != derived_reconciliation:
            blockers.append("reconciliation_report")
    if blockers:
        raise ExecutionPolicyError(
            "execution policy replay invalid: " + ", ".join(sorted(set(blockers)))
        )


def _derived_source_input_stream_sha256(replay: dict[str, Any]) -> str:
    hashes = [
        str(row.get("source_input_sha256") or "")
        for row in replay.get("decisions", [])
        if isinstance(row, dict)
    ]
    return canonical_payload_sha256(
        hashes,
        payload_schema_version=EXECUTION_POLICY_INPUT_SCHEMA_VERSION,
    )


def _derived_decision_stream_sha256(replay: dict[str, Any]) -> str:
    return canonical_payload_sha256(
        replay.get("decisions", []),
        payload_schema_version=EXECUTION_POLICY_DECISION_SCHEMA_VERSION,
    )


def _derived_attribution_stream_sha256(replay: dict[str, Any]) -> str:
    return canonical_payload_sha256(
        replay.get("decision_attribution", []),
        payload_schema_version=EXECUTION_POLICY_ATTRIBUTION_SCHEMA_VERSION,
    )


def _derived_risk_state_stream_sha256(replay: dict[str, Any]) -> str:
    return canonical_payload_sha256(
        replay.get("risk_budget_state", []),
        payload_schema_version=RISK_BUDGET_STATE_SCHEMA_VERSION,
    )


def _execution_output_sha256(replay: dict[str, Any]) -> str:
    payload = {
        field: replay.get(field, [])
        for field in (
            "decisions",
            "decision_attribution",
            "risk_budget_state",
            "intents",
            "fills",
            "positions",
            "ledger",
        )
    }
    return canonical_payload_sha256(
        payload,
        payload_schema_version=REPLAY_SCHEMA_VERSION,
    )


def _valid_decision(
    row: Any,
    *,
    sequence: int,
    candidate_id: str,
    source_model_hash: str,
    execution_policy_sha256: str,
) -> bool:
    if not isinstance(row, dict):
        return False
    expected_fields = {
        "schema_version",
        "sequence",
        "candidate_id",
        "market_id",
        "decision_ts",
        "source_model_hash",
        "execution_policy_hash",
        "source_input_sha256",
        "source_action_scores_sha256",
        "selected_source_score",
        "opportunity_window_id",
        "selected_action",
        "selected_side",
        "proposed_notional",
        "decision_effect",
        "reason_codes",
        "source_scores_mutated",
        "target_used_as_decision_input",
        "paper_only",
        "capital_at_risk",
        "decision_sha256",
    }
    if row.get("selected_action") != "NO_TRADE":
        expected_fields.add("released_notional")
    if set(row) != expected_fields:
        return False
    reason_codes = row.get("reason_codes")
    if (
        row.get("schema_version") != EXECUTION_POLICY_DECISION_SCHEMA_VERSION
        or row.get("sequence") != sequence
        or row.get("candidate_id") != candidate_id
        or row.get("source_model_hash") != source_model_hash
        or row.get("execution_policy_hash") != execution_policy_sha256
        or not _sha256(row.get("source_input_sha256"))
        or not _sha256(row.get("source_action_scores_sha256"))
        or not _finite_number(row.get("selected_source_score"))
        or not isinstance(reason_codes, list)
        or any(not isinstance(reason, str) or not reason for reason in reason_codes)
        or len(set(reason_codes)) != len(reason_codes)
        or row.get("source_scores_mutated") is not False
        or row.get("target_used_as_decision_input") is not False
        or row.get("paper_only") is not True
        or row.get("capital_at_risk") is not False
        or not isinstance(row.get("market_id"), str)
        or not isinstance(row.get("opportunity_window_id"), str)
        or _decision_timestamp(row.get("decision_ts")) != row.get("decision_ts")
    ):
        return False
    claimed = row.get("decision_sha256")
    if not _sha256(claimed) or claimed != _digest_without_field(
        row,
        field="decision_sha256",
        payload_schema_version=EXECUTION_POLICY_DECISION_SCHEMA_VERSION,
    ):
        return False
    action = row.get("selected_action")
    side = row.get("selected_side")
    notional = row.get("proposed_notional")
    effect = row.get("decision_effect")
    if action == "NO_TRADE":
        return (
            side == "NONE"
            and _finite_number(notional)
            and float(notional) == 0.0
            and effect == "rejected"
            and bool(reason_codes)
        )
    expected_side = "UP" if isinstance(action, str) and action.startswith("BUY_UP") else "DOWN"
    released = row.get("released_notional")
    return (
        action in BUY_ACTIONS
        and side == expected_side
        and _finite_number(notional)
        and float(notional) > 0.0
        and effect in {"accepted", "replaced"}
        and _finite_number(released)
        and float(released) >= 0.0
        and ((effect == "accepted" and float(released) == 0.0) or effect == "replaced")
        and bool(reason_codes)
    )


def _valid_attribution(row: Any, *, decision: Any, sequence: int) -> bool:
    if not isinstance(row, dict) or not isinstance(decision, dict):
        return False
    if set(row) != {
        "schema_version",
        "sequence",
        "decision_sha256",
        "market_id",
        "decision_ts",
        "candidate_id",
        "source_model_hash",
        "execution_policy_hash",
        "source_input_sha256",
        "effect",
        "rule_results",
        "selected_action",
        "selected_side",
        "sized_notional",
        "selected_source_score",
        "source_action_scores_sha256",
        "source_scores_mutated",
        "kill_switch_active",
        "attribution_sha256",
    }:
        return False
    claimed = row.get("attribution_sha256")
    expected_pairs = {
        "sequence": sequence,
        "decision_sha256": decision.get("decision_sha256"),
        "market_id": decision.get("market_id"),
        "decision_ts": decision.get("decision_ts"),
        "candidate_id": decision.get("candidate_id"),
        "source_model_hash": decision.get("source_model_hash"),
        "execution_policy_hash": decision.get("execution_policy_hash"),
        "source_input_sha256": decision.get("source_input_sha256"),
        "effect": decision.get("decision_effect"),
        "rule_results": decision.get("reason_codes"),
        "selected_action": decision.get("selected_action"),
        "selected_side": decision.get("selected_side"),
        "sized_notional": decision.get("proposed_notional"),
        "selected_source_score": decision.get("selected_source_score"),
        "source_action_scores_sha256": decision.get("source_action_scores_sha256"),
    }
    return (
        row.get("schema_version") == EXECUTION_POLICY_ATTRIBUTION_SCHEMA_VERSION
        and all(row.get(field) == value for field, value in expected_pairs.items())
        and row.get("source_scores_mutated") is False
        and (
            row.get("kill_switch_active") is True
            or row.get("kill_switch_active") is False
            or row.get("kill_switch_active") is None
        )
        and _sha256(claimed)
        and claimed
        == _digest_without_field(
            row,
            field="attribution_sha256",
            payload_schema_version=EXECUTION_POLICY_ATTRIBUTION_SCHEMA_VERSION,
        )
    )


def _valid_risk_state_row(
    row: Any,
    *,
    decision: Any,
    sequence: int,
    execution_policy_sha256: str,
    previous_after: dict[str, Any],
    previous_risk_sha256: str,
) -> bool:
    if not isinstance(row, dict) or not isinstance(decision, dict):
        return False
    if set(row) != {
        "schema_version",
        "sequence",
        "market_id",
        "decision_ts",
        "execution_policy_hash",
        "decision_sha256",
        "source_input_sha256",
        "previous_risk_budget_state_sha256",
        "before",
        "after",
        "risk_budget_state_sha256",
    }:
        return False
    claimed = row.get("risk_budget_state_sha256")
    before = row.get("before")
    after = row.get("after")
    return (
        row.get("schema_version") == RISK_BUDGET_STATE_SCHEMA_VERSION
        and row.get("sequence") == sequence
        and row.get("market_id") == decision.get("market_id")
        and row.get("decision_ts") == decision.get("decision_ts")
        and row.get("execution_policy_hash") == execution_policy_sha256
        and row.get("decision_sha256") == decision.get("decision_sha256")
        and row.get("source_input_sha256") == decision.get("source_input_sha256")
        and row.get("previous_risk_budget_state_sha256") == previous_risk_sha256
        and before == previous_after
        and _valid_state_snapshot(before)
        and _valid_state_snapshot(after)
        and _state_change_matches_decision(
            before=before,
            after=after,
            decision=decision,
        )
        and _sha256(claimed)
        and claimed
        == _digest_without_field(
            row,
            field="risk_budget_state_sha256",
            payload_schema_version=RISK_BUDGET_STATE_SCHEMA_VERSION,
        )
    )


def _valid_state_snapshot(snapshot: Any) -> bool:
    if not isinstance(snapshot, dict) or set(snapshot) != {
        "total_exposure",
        "side_exposure",
        "market_exposure",
        "position_count",
        "positions",
        "accepted_by_window",
        "last_action_ts_by_market",
    }:
        return False
    positions = snapshot.get("positions")
    if not isinstance(positions, list) or any(
        not _valid_position(position) for position in positions
    ):
        return False
    market_ids = [str(position["market_id"]) for position in positions]
    if market_ids != sorted(market_ids) or len(set(market_ids)) != len(market_ids):
        return False
    if snapshot.get("position_count") != len(positions):
        return False
    market_exposure = _sum_notional(positions, "market_id", value_field="notional")
    side_exposure = _sum_notional(positions, "side", value_field="notional")
    if not _numeric_maps_equal(snapshot.get("market_exposure"), market_exposure):
        return False
    if not _numeric_maps_equal(snapshot.get("side_exposure"), side_exposure):
        return False
    if not _finite_number(snapshot.get("total_exposure")) or not _close(
        float(snapshot["total_exposure"]), sum(market_exposure.values())
    ):
        return False
    accepted = snapshot.get("accepted_by_window")
    last_action = snapshot.get("last_action_ts_by_market")
    return (
        isinstance(accepted, dict)
        and all(
            isinstance(key, str)
            and bool(key)
            and isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0
            for key, value in accepted.items()
        )
        and list(accepted) == sorted(accepted)
        and isinstance(last_action, dict)
        and all(
            isinstance(key, str)
            and bool(key)
            and isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0
            for key, value in last_action.items()
        )
        and list(last_action) == sorted(last_action)
    )


def _state_change_matches_decision(
    *,
    before: dict[str, Any],
    after: dict[str, Any],
    decision: dict[str, Any],
) -> bool:
    effect = decision.get("decision_effect")
    if effect == "rejected":
        return before == after
    market_id = str(decision.get("market_id") or "")
    window_id = str(decision.get("opportunity_window_id") or "")
    if not market_id or not window_id:
        return False
    before_positions = {str(row["market_id"]): row for row in before.get("positions", [])}
    after_positions = {str(row["market_id"]): row for row in after.get("positions", [])}
    expected_position = {
        "market_id": market_id,
        "action": decision.get("selected_action"),
        "side": decision.get("selected_side"),
        "notional": decision.get("proposed_notional"),
        "source_score": None,
        "opened_ts": decision.get("decision_ts"),
    }
    actual_position = after_positions.get(market_id)
    if actual_position is None:
        return False
    for field in ("market_id", "action", "side", "notional", "opened_ts"):
        if actual_position.get(field) != expected_position[field]:
            return False
    if not _close(
        float(actual_position["source_score"]),
        float(decision.get("selected_source_score")),
    ):
        return False
    expected_before_markets = set(before_positions)
    expected_after_markets = set(after_positions)
    if effect == "accepted":
        if market_id in before_positions or expected_after_markets != (
            expected_before_markets | {market_id}
        ):
            return False
    elif effect == "replaced":
        if market_id not in before_positions or expected_after_markets != expected_before_markets:
            return False
        if not _close(
            float(before_positions[market_id]["notional"]),
            float(decision.get("released_notional")),
        ):
            return False
    else:
        return False
    if any(
        before_positions.get(key) != after_positions.get(key)
        for key in expected_before_markets - {market_id}
    ):
        return False
    expected_accepted = dict(before.get("accepted_by_window") or {})
    expected_accepted[window_id] = int(expected_accepted.get(window_id, 0)) + 1
    if after.get("accepted_by_window") != dict(sorted(expected_accepted.items())):
        return False
    expected_last_action = dict(before.get("last_action_ts_by_market") or {})
    expected_last_action[market_id] = int(decision["decision_ts"])
    return after.get("last_action_ts_by_market") == dict(sorted(expected_last_action.items()))


def _empty_state_snapshot() -> dict[str, Any]:
    return {
        "total_exposure": 0,
        "side_exposure": {},
        "market_exposure": {},
        "position_count": 0,
        "positions": [],
        "accepted_by_window": {},
        "last_action_ts_by_market": {},
    }


def _valid_position(row: Any) -> bool:
    if not isinstance(row, dict) or set(row) != {
        "market_id",
        "action",
        "side",
        "notional",
        "source_score",
        "opened_ts",
    }:
        return False
    action = row.get("action")
    expected_side = "UP" if isinstance(action, str) and action.startswith("BUY_UP") else "DOWN"
    return (
        isinstance(row.get("market_id"), str)
        and bool(row["market_id"])
        and action in BUY_ACTIONS
        and row.get("side") == expected_side
        and _finite_number(row.get("notional"))
        and float(row["notional"]) > 0.0
        and _finite_number(row.get("source_score"))
        and isinstance(row.get("opened_ts"), int)
        and not isinstance(row.get("opened_ts"), bool)
        and int(row["opened_ts"]) >= 0
    )


def _valid_transition_row(row: Any, *, id_field: str) -> bool:
    if not isinstance(row, dict):
        return False
    expected_fields = {
        id_field,
        "market_id",
        "side",
        "notional_delta",
        "sequence",
        "decision_sha256",
        "transition_type",
        "paper_only",
        "capital_at_risk",
    }
    if id_field == "fill_id":
        expected_fields.add("intent_id")
    elif id_field == "ledger_entry_id":
        expected_fields.add("fill_id")
    if set(row) != expected_fields:
        return False
    transition_type = row.get("transition_type")
    delta = row.get("notional_delta")
    return (
        isinstance(row.get(id_field), str)
        and bool(row[id_field])
        and isinstance(row.get("market_id"), str)
        and bool(row["market_id"])
        and row.get("side") in {"UP", "DOWN"}
        and _finite_number(delta)
        and float(delta) != 0.0
        and transition_type in {"entry", "replacement_exit", "replacement_entry"}
        and (
            (transition_type == "replacement_exit" and float(delta) < 0.0)
            or (transition_type != "replacement_exit" and float(delta) > 0.0)
        )
        and isinstance(row.get("sequence"), int)
        and not isinstance(row.get("sequence"), bool)
        and int(row["sequence"]) > 0
        and _sha256(row.get("decision_sha256"))
        and row.get("paper_only") is True
        and row.get("capital_at_risk") is False
    )


def _transition_rows_match(row: Any, parent: Any) -> bool:
    if not isinstance(row, dict) or not isinstance(parent, dict):
        return False
    return (
        all(
            row.get(field) == parent.get(field)
            for field in (
                "market_id",
                "side",
                "sequence",
                "decision_sha256",
                "transition_type",
                "paper_only",
                "capital_at_risk",
            )
        )
        and _finite_number(row.get("notional_delta"))
        and _finite_number(parent.get("notional_delta"))
        and _close(float(row["notional_delta"]), float(parent["notional_delta"]))
    )


def _decision_transition_checks(
    *,
    decisions: list[dict[str, Any]],
    ledger: list[dict[str, Any]],
    candidate_id: str,
) -> bool:
    if any(
        not isinstance(row, dict)
        or not isinstance(row.get("sequence"), int)
        or row["sequence"] < 1
        or row["sequence"] > len(decisions)
        for row in ledger
    ):
        return False
    by_sequence: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in ledger:
        by_sequence[int(row["sequence"])].append(row)
    for sequence, decision in enumerate(decisions, start=1):
        if not isinstance(decision, dict):
            return False
        rows = by_sequence.get(sequence, [])
        if any(row.get("decision_sha256") != decision.get("decision_sha256") for row in rows):
            return False
        effect = decision.get("decision_effect")
        types = Counter(str(row.get("transition_type")) for row in rows)
        if effect == "rejected":
            if rows:
                return False
            continue
        if effect == "accepted":
            if types != Counter({"entry": 1}):
                return False
        elif effect == "replaced":
            if types != Counter({"replacement_exit": 1, "replacement_entry": 1}):
                return False
        else:
            return False
        entries = [row for row in rows if row.get("transition_type") != "replacement_exit"]
        if (
            len(entries) != 1
            or entries[0].get("market_id") != decision.get("market_id")
            or entries[0].get("side") != decision.get("selected_side")
            or not _close(
                float(entries[0]["notional_delta"]),
                float(decision.get("proposed_notional")),
            )
        ):
            return False
        exits = [row for row in rows if row.get("transition_type") == "replacement_exit"]
        if exits and (
            exits[0].get("market_id") != decision.get("market_id")
            or not _close(
                -float(exits[0]["notional_delta"]),
                float(decision.get("released_notional")),
            )
        ):
            return False
    return all(
        str(row.get("ledger_entry_id"))
        == f"ledger:{candidate_id}:{row.get('sequence')}:{row.get('transition_type')}"
        for row in ledger
    )


def _risk_state_transition_checks(
    *,
    risk_states: list[dict[str, Any]],
    ledger: list[dict[str, Any]],
) -> bool:
    ledger_by_sequence: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in ledger:
        if isinstance(row, dict) and isinstance(row.get("sequence"), int):
            ledger_by_sequence[int(row["sequence"])].append(row)
    for sequence, state in enumerate(risk_states, start=1):
        if not isinstance(state, dict):
            return False
        before = state.get("before")
        after = state.get("after")
        if not isinstance(before, dict) or not isinstance(after, dict):
            return False
        rows = ledger_by_sequence.get(sequence, [])
        market_delta = _sum_notional(rows, "market_id")
        side_delta = _sum_notional(rows, "side")
        before_market = before.get("market_exposure")
        after_market = after.get("market_exposure")
        before_side = before.get("side_exposure")
        after_side = after.get("side_exposure")
        if not all(
            isinstance(value, dict)
            for value in (before_market, after_market, before_side, after_side)
        ):
            return False
        if not _numeric_maps_equal(
            _numeric_map_difference(after_market, before_market), market_delta
        ):
            return False
        if not _numeric_maps_equal(_numeric_map_difference(after_side, before_side), side_delta):
            return False
        before_total = before.get("total_exposure")
        after_total = after.get("total_exposure")
        if (
            not _finite_number(before_total)
            or not _finite_number(after_total)
            or not _close(
                float(after_total) - float(before_total),
                sum(market_delta.values()),
            )
        ):
            return False
    return True


def _sum_notional(
    rows: list[dict[str, Any]],
    key_field: str,
    *,
    value_field: str = "notional_delta",
) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for row in rows:
        if not isinstance(row, dict):
            totals["__invalid__"] = math.nan
            continue
        key = row.get(key_field)
        value = row.get(value_field)
        if not isinstance(key, str) or not key or not _finite_number(value):
            totals["__invalid__"] = math.nan
            continue
        totals[key] += float(value)
    return {
        key: value
        for key, value in sorted(totals.items())
        if not _close(value, 0.0) or key == "__invalid__"
    }


def _numeric_map_difference(
    minuend: dict[str, Any],
    subtrahend: dict[str, Any],
) -> dict[str, float]:
    output: dict[str, float] = {}
    for key in set(minuend) | set(subtrahend):
        left = minuend.get(key, 0.0)
        right = subtrahend.get(key, 0.0)
        if not _finite_number(left) or not _finite_number(right):
            return {"__invalid__": math.nan}
        delta = float(left) - float(right)
        if not _close(delta, 0.0):
            output[str(key)] = delta
    return dict(sorted(output.items()))


def _numeric_maps_equal(left: Any, right: Any) -> bool:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    if set(left) != set(right):
        return False
    return all(
        _finite_number(left[key])
        and _finite_number(right[key])
        and _close(float(left[key]), float(right[key]))
        for key in left
    )


def _close(left: float, right: float) -> bool:
    return (
        math.isfinite(left)
        and math.isfinite(right)
        and math.isclose(
            left,
            right,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(float(value))
    )


def _structural_equal(left: Any, right: Any) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            _structural_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _structural_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    if isinstance(left, float) and isinstance(right, float):
        return (math.isnan(left) and math.isnan(right)) or left == right
    return type(left) is type(right) and left == right


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _decision_timestamp(value: Any) -> int:
    return int(value) if _valid_decision_timestamp(value) else 0


def _valid_decision_timestamp(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _digest_without_field(
    row: dict[str, Any],
    *,
    field: str,
    payload_schema_version: str,
) -> str | None:
    payload = copy.deepcopy(row)
    payload.pop(field, None)
    try:
        return canonical_payload_sha256(
            payload,
            payload_schema_version=payload_schema_version,
        )
    except (TypeError, ValueError, OverflowError):
        return None


def _safe_canonical_hash(
    payload: Any,
    *,
    payload_schema_version: str,
) -> str:
    try:
        return canonical_payload_sha256(
            payload,
            payload_schema_version=payload_schema_version,
        )
    except (TypeError, ValueError, OverflowError):
        return canonical_payload_sha256(
            _hash_projection(payload),
            payload_schema_version=f"{payload_schema_version}-invalid-projection",
        )


def _hash_projection(value: Any) -> dict[str, Any]:
    """Build a deterministic, target-field-neutral projection for invalid input."""

    if value is None:
        return {"kind": "none"}
    if isinstance(value, bool):
        return {"kind": "bool", "payload": value}
    if isinstance(value, int):
        return {"kind": "int", "payload": str(value)}
    if isinstance(value, float):
        if math.isnan(value):
            payload = "nan"
        elif math.isinf(value):
            payload = "positive_infinity" if value > 0 else "negative_infinity"
        else:
            payload = value
        return {"kind": "float", "payload": payload}
    if isinstance(value, str):
        return {"kind": "string", "payload": value}
    if isinstance(value, bytes):
        return {"kind": "bytes", "payload": value.hex()}
    if isinstance(value, dict):
        entries = [
            {
                "projected_key": _hash_projection(key),
                "projected_value": _hash_projection(item),
            }
            for key, item in value.items()
        ]
        entries.sort(
            key=lambda entry: canonical_payload_sha256(
                entry["projected_key"],
                payload_schema_version="bigan-v8-invalid-input-projected-key-v1",
            )
        )
        return {"kind": "mapping", "entries": entries}
    if isinstance(value, (list, tuple)):
        return {
            "kind": "list" if isinstance(value, list) else "tuple",
            "items": [_hash_projection(item) for item in value],
        }
    return {
        "kind": f"unsupported:{type(value).__module__}.{type(value).__qualname__}",
        "payload": repr(value),
    }


def _decide(
    *,
    source_input: dict[str, Any],
    policy: dict[str, Any],
    policy_hash: str,
    source_model_hash: str,
    state: dict[str, Any],
    sequence: int,
    source_input_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    market_id = str(source_input.get("market_id") or "")
    decision_ts = _decision_timestamp(source_input.get("decision_ts"))
    base = {
        "schema_version": EXECUTION_POLICY_DECISION_SCHEMA_VERSION,
        "sequence": sequence,
        "candidate_id": policy["candidate_id"],
        "market_id": market_id,
        "decision_ts": decision_ts,
        "source_model_hash": source_model_hash,
        "execution_policy_hash": policy_hash,
        "source_input_sha256": source_input_sha256,
        "source_action_scores_sha256": "",
        "selected_source_score": 0.0,
        "opportunity_window_id": str(source_input.get("opportunity_window_id") or ""),
        "selected_action": "NO_TRADE",
        "selected_side": "NONE",
        "proposed_notional": 0.0,
        "decision_effect": "rejected",
        "reason_codes": [],
        "source_scores_mutated": False,
        "target_used_as_decision_input": False,
        "paper_only": True,
        "capital_at_risk": False,
    }
    reasons = _input_blockers(
        source_input,
        policy=policy,
        source_model_hash=source_model_hash,
    )
    scores = source_input.get("source_action_scores")
    if isinstance(scores, dict):
        base["source_action_scores_sha256"] = _safe_canonical_hash(
            scores,
            payload_schema_version="bigan-v8-source-action-scores-v1",
        )
    else:
        base["source_action_scores_sha256"] = "0" * 64
    if reasons:
        return _finalize_no_trade(base, reasons, source_input)
    assert isinstance(scores, dict)
    ranked = sorted(
        ((str(action), float(score)) for action, score in scores.items() if action != "NO_TRADE"),
        key=lambda item: (-item[1], item[0]),
    )
    action, score = ranked[0]
    side = "UP" if action.startswith("BUY_UP") else "DOWN"
    constraints = policy["constraints"]
    if score < float(constraints["minimum_source_score"]):
        return _finalize_no_trade(base, ["minimum_source_score_not_met"], source_input)
    if float(source_input["uncertainty"]) > float(constraints["maximum_uncertainty"]):
        return _finalize_no_trade(base, ["maximum_uncertainty_exceeded"], source_input)
    if float(source_input["fill_quality_score"]) < float(constraints["minimum_fill_quality_score"]):
        return _finalize_no_trade(base, ["fill_quality_below_minimum"], source_input)
    if float(source_input["provider_health_score"]) < float(
        constraints["minimum_provider_health_score"]
    ):
        return _finalize_no_trade(base, ["provider_health_below_minimum"], source_input)
    window_id = str(source_input["opportunity_window_id"])
    if int(state["accepted_by_window"][window_id]) >= int(
        constraints["maximum_opportunities_per_window"]
    ):
        return _finalize_no_trade(base, ["opportunity_budget_exhausted"], source_input)
    existing: _Position | None = state["positions"].get(market_id)
    last_ts = state["last_action_ts_by_market"].get(market_id)
    if existing is not None and existing.action == action:
        return _finalize_no_trade(base, ["duplicate_position"], source_input)
    if last_ts is not None and decision_ts - int(last_ts) < int(constraints["reentry_cooldown_ms"]):
        return _finalize_no_trade(base, ["reentry_cooldown_active"], source_input)
    transitions: list[dict[str, Any]] = []
    released = 0.0
    effect = "accepted"
    if existing is not None:
        if constraints["replacement_enabled"] is not True:
            return _finalize_no_trade(base, ["replacement_disabled"], source_input)
        if score < existing.source_score + float(constraints["replacement_minimum_score_uplift"]):
            return _finalize_no_trade(base, ["replacement_score_uplift_not_met"], source_input)
        released = existing.notional
        transitions.append(
            {
                "transition_type": "replacement_exit",
                "market_id": market_id,
                "side": existing.side,
                "notional_delta": -existing.notional,
            }
        )
        del state["positions"][market_id]
        effect = "replaced"
    exposure = _exposure(state["positions"])
    desired = float(constraints["paper_notional_per_position"])
    remaining_global = float(constraints["global_exposure_cap"]) - exposure["total"]
    remaining_side = float(constraints["per_side_exposure_cap"]) - exposure["side"].get(side, 0.0)
    remaining_market = float(constraints["per_market_exposure_cap"])
    notional = min(desired, remaining_global, remaining_side, remaining_market)
    if _close(notional, desired):
        notional = desired
    if notional <= 0.0 or (
        constraints["partial_sizing_allowed"] is not True
        and notional < desired
        and not _close(notional, desired)
    ):
        if existing is not None:
            state["positions"][market_id] = existing
            transitions.clear()
        return _finalize_no_trade(base, ["exposure_budget_exhausted"], source_input)
    position = _Position(
        market_id=market_id,
        action=action,
        side=side,
        notional=notional,
        source_score=score,
        opened_ts=decision_ts,
    )
    state["positions"][market_id] = position
    state["accepted_by_window"][window_id] += 1
    state["last_action_ts_by_market"][market_id] = decision_ts
    transitions.append(
        {
            "transition_type": "replacement_entry" if effect == "replaced" else "entry",
            "market_id": market_id,
            "side": side,
            "notional_delta": notional,
        }
    )
    base.update(
        {
            "selected_action": action,
            "selected_side": side,
            "selected_source_score": score,
            "proposed_notional": notional,
            "decision_effect": effect,
            "reason_codes": [
                "source_score_ranked",
                "all_execution_constraints_passed",
                "position_replaced" if effect == "replaced" else "position_accepted",
            ],
            "released_notional": released,
        }
    )
    return _finalize_decision(base, source_input, transitions)


def _input_blockers(
    source_input: dict[str, Any],
    *,
    policy: dict[str, Any],
    source_model_hash: str,
) -> list[str]:
    reasons = []
    input_keys = set(source_input)
    if REQUIRED_DECISION_TIME_INPUTS - input_keys:
        reasons.append("required_execution_input_missing")
    if input_keys - REQUIRED_DECISION_TIME_INPUTS:
        reasons.append("unsupported_execution_input_present")
    if any(
        any(token in str(key).lower() for token in FORBIDDEN_INPUT_TOKENS) for key in source_input
    ):
        reasons.append("forbidden_target_or_future_input_present")
    if source_input.get("source_model_hash") != source_model_hash:
        reasons.append("source_model_hash_mismatch")
    if (
        not isinstance(source_input.get("market_id"), str)
        or not str(source_input.get("market_id") or "").strip()
    ):
        reasons.append("market_id_invalid")
    if not _valid_decision_timestamp(source_input.get("decision_ts")):
        reasons.append("decision_ts_invalid")
    if (
        not isinstance(source_input.get("opportunity_window_id"), str)
        or not str(source_input.get("opportunity_window_id") or "").strip()
    ):
        reasons.append("opportunity_window_id_invalid")
    if source_input.get("provider_features_complete") is not True:
        reasons.append("provider_features_incomplete")
    if source_input.get("kill_switch_active") is not False:
        reasons.append("policy_kill_switch_active_or_missing")
    scores = source_input.get("source_action_scores")
    if not isinstance(scores, dict) or not scores:
        reasons.append("source_action_scores_missing")
    elif (
        set(scores) - ALLOWED_ACTIONS
        or not set(scores).intersection(BUY_ACTIONS)
        or any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            for value in scores.values()
        )
    ):
        reasons.append("source_action_scores_invalid")
    for field in ("uncertainty", "fill_quality_score", "provider_health_score"):
        value = source_input.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
        ):
            reasons.append(f"{field}_invalid")
    if policy.get("constraints", {}).get("paper_notional_per_position", 0) <= 0:
        reasons.append("policy_notional_invalid")
    return sorted(set(reasons))


def _finalize_no_trade(
    base: dict[str, Any],
    reasons: list[str],
    source_input: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    base["reason_codes"] = sorted(set(reasons))
    return _finalize_decision(base, source_input, [])


def _finalize_decision(
    base: dict[str, Any],
    source_input: dict[str, Any],
    transitions: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    decision = copy.deepcopy(base)
    decision["decision_sha256"] = canonical_payload_sha256(
        decision,
        payload_schema_version=EXECUTION_POLICY_DECISION_SCHEMA_VERSION,
    )
    attribution = {
        "schema_version": EXECUTION_POLICY_ATTRIBUTION_SCHEMA_VERSION,
        "sequence": decision["sequence"],
        "decision_sha256": decision["decision_sha256"],
        "market_id": decision["market_id"],
        "decision_ts": decision["decision_ts"],
        "candidate_id": decision["candidate_id"],
        "source_model_hash": decision["source_model_hash"],
        "execution_policy_hash": decision["execution_policy_hash"],
        "source_input_sha256": decision["source_input_sha256"],
        "effect": decision["decision_effect"],
        "rule_results": list(decision["reason_codes"]),
        "selected_action": decision["selected_action"],
        "selected_side": decision["selected_side"],
        "sized_notional": decision["proposed_notional"],
        "selected_source_score": decision["selected_source_score"],
        "source_action_scores_sha256": decision["source_action_scores_sha256"],
        "source_scores_mutated": False,
        "kill_switch_active": source_input.get("kill_switch_active"),
    }
    attribution["attribution_sha256"] = canonical_payload_sha256(
        attribution,
        payload_schema_version=str(attribution["schema_version"]),
    )
    return decision, attribution, transitions


def _state_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    exposure = _exposure(state["positions"])
    return {
        "total_exposure": exposure["total"],
        "side_exposure": exposure["side"],
        "market_exposure": exposure["market"],
        "position_count": len(state["positions"]),
        "positions": [position.to_dict() for _, position in sorted(state["positions"].items())],
        "accepted_by_window": dict(sorted(state["accepted_by_window"].items())),
        "last_action_ts_by_market": dict(sorted(state["last_action_ts_by_market"].items())),
    }


def _exposure(positions: dict[str, _Position]) -> dict[str, Any]:
    side: dict[str, float] = defaultdict(float)
    market: dict[str, float] = {}
    for market_id, position in positions.items():
        side[position.side] += position.notional
        market[market_id] = position.notional
    return {
        "total": sum(position.notional for position in positions.values()),
        "side": dict(sorted(side.items())),
        "market": dict(sorted(market.items())),
    }


__all__ = [
    "ALLOWED_ACTIONS",
    "EXECUTION_POLICY_CONTRACT_SCHEMA_VERSION",
    "EXECUTION_POLICY_DECISION_SCHEMA_VERSION",
    "EXECUTION_POLICY_FIXTURE_SCHEMA_VERSION",
    "ExecutionPolicyError",
    "build_policy_safety_report",
    "build_replay_parity_report",
    "execution_policy_hash",
    "run_execution_policy_replay",
    "validate_execution_policy_contract",
    "validate_execution_policy_fixture",
    "validate_execution_policy_future_validation_protocol",
    "validate_execution_policy_replay",
    "validate_policy_candidate_manifest",
    "validate_policy_reconciliation",
    "validate_source_execution_compatibility",
]
