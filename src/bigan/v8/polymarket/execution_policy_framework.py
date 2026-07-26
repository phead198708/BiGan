"""Bounded, deterministic execution-policy framework for v8 challenge models."""

from __future__ import annotations

import copy
import math
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
BUY_ACTIONS = (
    "BUY_UP_HOLD_TO_SETTLEMENT",
    "BUY_DOWN_HOLD_TO_SETTLEMENT",
    "BUY_UP_SELL_BEFORE_CLOSE",
    "BUY_DOWN_SELL_BEFORE_CLOSE",
)
ALLOWED_ACTIONS = frozenset((*BUY_ACTIONS, "NO_TRADE"))
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
    blockers: list[str] = []
    if contract.get("schema_version") != EXECUTION_POLICY_CONTRACT_SCHEMA_VERSION:
        blockers.append("schema_version")
    if contract.get("source_model_and_execution_policy_independently_versioned") is not True:
        blockers.append("independent_versioning")
    if contract.get("source_scores_mutation_allowed") is not False:
        blockers.append("source_score_mutation")
    if contract.get("outcome_driven_policy_search_allowed") is not False:
        blockers.append("outcome_search")
    if contract.get("unsupported_or_missing_inputs") != "fail_closed_no_trade":
        blockers.append("missing_input_behavior")
    if any(contract.get("safety", {}).values()):
        blockers.append("safety")
    if blockers:
        raise ExecutionPolicyError(
            "execution policy contract invalid: " + ", ".join(sorted(blockers))
        )


def execution_policy_hash(policy: dict[str, Any]) -> str:
    if policy.get("schema_version") != EXECUTION_POLICY_FIXTURE_SCHEMA_VERSION:
        raise ExecutionPolicyError("execution policy fixture schema invalid")
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
    blockers: list[str] = []
    if compatibility_manifest.get("source_model_hash") != source_model_hash:
        blockers.append("source_model_hash")
    if execution_policy_hash(policy) not in set(
        compatibility_manifest.get("allowed_execution_policy_hashes") or []
    ):
        blockers.append("execution_policy_hash")
    required = set(compatibility_manifest.get("required_decision_time_inputs") or [])
    if not required:
        blockers.append("required_inputs")
    if compatibility_manifest.get("source_action_scores_are_read_only") is not True:
        blockers.append("source_scores_read_only")
    if compatibility_manifest.get("target_or_outcome_inputs_allowed") is not False:
        blockers.append("target_inputs")
    if blockers:
        raise ExecutionPolicyError(
            "source/execution compatibility invalid: " + ", ".join(sorted(blockers))
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
    policy_hash = execution_policy_hash(policy)
    source_model_hash = str(compatibility_manifest["source_model_hash"])
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
    for sequence, source_input in enumerate(inputs, start=1):
        decision_ts = int(source_input.get("decision_ts") or -1)
        if decision_ts < previous_decision_ts:
            raise ExecutionPolicyError("policy replay inputs must be chronological")
        previous_decision_ts = decision_ts
        before = _state_snapshot(state)
        decision, attribution, transitions = _decide(
            source_input=source_input,
            policy=policy,
            policy_hash=policy_hash,
            source_model_hash=source_model_hash,
            state=state,
            sequence=sequence,
        )
        decisions.append(decision)
        attributions.append(attribution)
        for transition in transitions:
            intent_id = (
                f"{policy['candidate_id']}:{sequence}:{transition['transition_type']}"
            )
            intent = {
                "intent_id": intent_id,
                "market_id": transition["market_id"],
                "side": transition["side"],
                "notional_delta": transition["notional_delta"],
                "paper_only": True,
                "capital_at_risk": False,
            }
            fill = {
                "fill_id": f"fill:{intent_id}",
                "intent_id": intent_id,
                "market_id": transition["market_id"],
                "side": transition["side"],
                "notional_delta": transition["notional_delta"],
                "paper_only": True,
                "capital_at_risk": False,
            }
            entry = {
                "ledger_entry_id": f"ledger:{intent_id}",
                "fill_id": fill["fill_id"],
                "market_id": transition["market_id"],
                "side": transition["side"],
                "notional_delta": transition["notional_delta"],
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
            "before": before,
            "after": after,
        }
        risk_row["risk_budget_state_sha256"] = canonical_payload_sha256(
            risk_row,
            payload_schema_version=RISK_BUDGET_STATE_SCHEMA_VERSION,
        )
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
        "positions": [
            position.to_dict()
            for _, position in sorted(state["positions"].items())
        ],
        "ledger": ledger,
        "source_scores_mutated": False,
        "outcomes_settlement_pnl_future_returns_or_oracle_actions_used": False,
        "paper_only": True,
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
    return replay


def build_replay_parity_report(
    *,
    offline_replay: dict[str, Any],
    paper_runtime: dict[str, Any],
) -> dict[str, Any]:
    """Require exact policy decision and state parity across both surfaces."""

    checks = {
        "source_model_hash_match": offline_replay.get("source_model_hash")
        == paper_runtime.get("source_model_hash"),
        "execution_policy_hash_match": offline_replay.get("execution_policy_hash")
        == paper_runtime.get("execution_policy_hash"),
        "decision_stream_sha256_match": offline_replay.get("decision_stream_sha256")
        == paper_runtime.get("decision_stream_sha256"),
        "risk_state_stream_sha256_match": offline_replay.get("risk_state_stream_sha256")
        == paper_runtime.get("risk_state_stream_sha256"),
        "offline_reconciliation_passed": offline_replay.get(
            "reconciliation_report", {}
        ).get("passed")
        is True,
        "paper_reconciliation_passed": paper_runtime.get(
            "reconciliation_report", {}
        ).get("passed")
        is True,
    }
    return {
        "schema_version": "bigan-v8-execution-policy-replay-parity-v1",
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
    decisions = list(replay.get("decisions") or [])
    reason_counts = Counter(
        reason
        for decision in decisions
        for reason in decision.get("reason_codes", [])
    )
    checks = {
        "source_scores_immutable": replay.get("source_scores_mutated") is False,
        "no_target_or_outcome_inputs": replay.get(
            "outcomes_settlement_pnl_future_returns_or_oracle_actions_used"
        )
        is False,
        "paper_only": replay.get("paper_only") is True,
        "capital_at_risk_false": replay.get("capital_at_risk") is False,
        "reconciliation_passed": replay.get("reconciliation_report", {}).get("passed")
        is True,
        "all_no_trade_has_reason": all(
            decision.get("selected_action") != "NO_TRADE"
            or bool(decision.get("reason_codes"))
            for decision in decisions
        ),
    }
    return {
        "schema_version": "bigan-v8-execution-policy-safety-report-v1",
        "checks": checks,
        "passed": all(checks.values()),
        "reason_counts": dict(sorted(reason_counts.items())),
        "policy_kill_switch_separate_from_source_scores": True,
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
    intent_by_id = {str(row.get("intent_id")): row for row in intents}
    fill_by_id = {str(row.get("fill_id")): row for row in fills}
    checks = {
        "intent_ids_unique": len(intent_by_id) == len(intents),
        "fill_ids_unique": len(fill_by_id) == len(fills),
        "one_fill_per_intent": len(fills) == len(intents)
        and {str(row.get("intent_id")) for row in fills} == set(intent_by_id),
        "one_ledger_entry_per_fill": len(ledger) == len(fills)
        and {str(row.get("fill_id")) for row in ledger} == set(fill_by_id),
        "intent_fill_notional_match": all(
            float(row["notional_delta"])
            == float(intent_by_id[str(row["intent_id"])]["notional_delta"])
            for row in fills
            if str(row.get("intent_id")) in intent_by_id
        ),
        "fill_ledger_notional_match": all(
            float(row["notional_delta"])
            == float(fill_by_id[str(row["fill_id"])]["notional_delta"])
            for row in ledger
            if str(row.get("fill_id")) in fill_by_id
        ),
        "ledger_exposure_reconciles_positions": math.isclose(
            sum(float(row["notional_delta"]) for row in ledger),
            sum(float(row["notional"]) for row in positions),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "paper_only": all(row.get("paper_only") is True for row in (*intents, *fills, *ledger)),
        "capital_at_risk_false": all(
            row.get("capital_at_risk") is False for row in (*intents, *fills, *ledger)
        ),
    }
    report = {
        "schema_version": "bigan-v8-execution-policy-reconciliation-v1",
        "checks": checks,
        "passed": all(checks.values()),
        "intent_count": len(intents),
        "fill_count": len(fills),
        "position_count": len(positions),
        "ledger_entry_count": len(ledger),
    }
    if not report["passed"]:
        raise ExecutionPolicyError("policy position/intent/fill/ledger reconciliation failed")
    return report


def _decide(
    *,
    source_input: dict[str, Any],
    policy: dict[str, Any],
    policy_hash: str,
    source_model_hash: str,
    state: dict[str, Any],
    sequence: int,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    market_id = str(source_input.get("market_id") or "")
    decision_ts = int(source_input.get("decision_ts") or -1)
    base = {
        "schema_version": EXECUTION_POLICY_DECISION_SCHEMA_VERSION,
        "sequence": sequence,
        "candidate_id": policy["candidate_id"],
        "market_id": market_id,
        "decision_ts": decision_ts,
        "source_model_hash": source_model_hash,
        "execution_policy_hash": policy_hash,
        "source_action_scores_sha256": "",
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
        base["source_action_scores_sha256"] = canonical_payload_sha256(
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
    if float(source_input["uncertainty"]) > float(
        constraints["maximum_uncertainty"]
    ):
        return _finalize_no_trade(base, ["maximum_uncertainty_exceeded"], source_input)
    if float(source_input["fill_quality_score"]) < float(
        constraints["minimum_fill_quality_score"]
    ):
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
    if (
        existing is None
        and last_ts is not None
        and decision_ts - int(last_ts) < int(constraints["reentry_cooldown_ms"])
    ):
        return _finalize_no_trade(base, ["reentry_cooldown_active"], source_input)
    transitions: list[dict[str, Any]] = []
    released = 0.0
    effect = "accepted"
    if existing is not None:
        if constraints["replacement_enabled"] is not True:
            return _finalize_no_trade(base, ["replacement_disabled"], source_input)
        if score < existing.source_score + float(
            constraints["replacement_minimum_score_uplift"]
        ):
            return _finalize_no_trade(
                base, ["replacement_score_uplift_not_met"], source_input
            )
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
    remaining_side = float(constraints["per_side_exposure_cap"]) - exposure["side"].get(
        side, 0.0
    )
    remaining_market = float(constraints["per_market_exposure_cap"])
    notional = min(desired, remaining_global, remaining_side, remaining_market)
    if notional <= 0.0 or (
        constraints["partial_sizing_allowed"] is not True and notional < desired
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
    required = {
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
    reasons = []
    if required - set(source_input):
        reasons.append("required_execution_input_missing")
    if any(
        any(token in str(key).lower() for token in FORBIDDEN_INPUT_TOKENS)
        for key in source_input
    ):
        reasons.append("forbidden_target_or_future_input_present")
    if source_input.get("source_model_hash") != source_model_hash:
        reasons.append("source_model_hash_mismatch")
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
        "schema_version": "bigan-v8-execution-policy-attribution-v1",
        "decision_sha256": decision["decision_sha256"],
        "market_id": decision["market_id"],
        "decision_ts": decision["decision_ts"],
        "candidate_id": decision["candidate_id"],
        "effect": decision["decision_effect"],
        "rule_results": list(decision["reason_codes"]),
        "selected_action": decision["selected_action"],
        "selected_side": decision["selected_side"],
        "sized_notional": decision["proposed_notional"],
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
        "positions": [
            position.to_dict()
            for _, position in sorted(state["positions"].items())
        ],
        "accepted_by_window": dict(sorted(state["accepted_by_window"].items())),
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
    "validate_policy_reconciliation",
    "validate_source_execution_compatibility",
]
