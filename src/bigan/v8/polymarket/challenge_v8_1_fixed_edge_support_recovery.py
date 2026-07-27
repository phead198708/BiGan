"""Preregistered slot-4 fixed-edge support recovery for v8.1."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from bigan.v8.polymarket.challenge_historical_development import SAFE_FALSES
from bigan.v8.polymarket.challenge_v8_1_entry_price_floor import (
    _base_market_close_ts,
    _integer,
    _side_for_action,
    _validate_base_decision,
)
from bigan.v8.polymarket.contracts import canonical_json_sha256

PROFILE_SCHEMA_VERSION = (
    "bigan-v8-challenge-v8-1-fixed-edge-support-recovery-profile-v1"
)
CANDIDATE_DECISION_SCHEMA_VERSION = (
    "bigan-v8-challenge-v8-1-fixed-edge-support-recovery-decision-v1"
)
CANDIDATE_ID = "v8_1_fixed_edge_support_recovery_0_025"
BASE_CANDIDATE_ID = "adaptive_support_controller_v8_1"
FIXED_EDGE_THRESHOLD = 0.025
DECLARED_POSITION_SIZE = 0.2
EXPECTED_LINEAGE = {
    "base_v8_1_guard_replay_sha256": (
        "7534b07c41d1a1f2afe3519e7eef9146c9b84bdda72f88a22b9d0351baa9ce2d"
    ),
    "base_v8_1_market_comparison_sha256": (
        "fce95987a10b160d7a7e6cdfd3842cc3e3b34dd138eb27093fb9f86b0a790eae"
    ),
    "exact_195_market_ids_sha256": (
        "fef9eda7b8dac138b88c75f96b010bd40953795b2bcf7424debf77a004e06883"
    ),
    "iteration_003_entry_semantic_sha256": (
        "abe1eb3b6e03530c15cb51326801e783454a0be4d5885edd7dcca8dab22779ae"
    ),
    "preregistration_commit": "f9ccb562596d72c2f466b503a588aa682ad43fc6",
    "preregistration_sha256": (
        "315865ab003f03c7abe98fb8126758dadcdb44e400a42f71ec419de3cc2cec12"
    ),
    "scale_invariance_governance_sha256": (
        "e8898ef5aa1c4b796109c0d03920d794842472bdfb271f9db7221a100bc8590f"
    ),
    "success_standard_v2_sha256": (
        "01b6d0c80cd9f54cf78523e556788788dd3ac6324dc1865d385f6f4cf2dcb9bb"
    ),
}


class ChallengeFixedEdgeSupportRecoveryError(ValueError):
    """Raised when the fixed-edge candidate cannot fail closed."""


def validate_fixed_edge_support_recovery_profile(
    profile: Mapping[str, Any],
) -> None:
    """Validate the exact single-hypothesis slot-4 policy."""

    checks = {
        "identity": profile.get("schema_version") == PROFILE_SCHEMA_VERSION
        and profile.get("candidate_id") == CANDIDATE_ID,
        "lineage": profile.get("lineage") == EXPECTED_LINEAGE,
        "policy": profile.get("policy")
        == {
            "adaptive_rolling_quantile_gate_used_for_final_selection": False,
            "base_candidate_id": BASE_CANDIDATE_ID,
            "base_controller_state_transition_changed": False,
            "fixed_edge_threshold_inclusive": FIXED_EDGE_THRESHOLD,
            "fixed_edge_threshold_source": "existing_v8_1_fixed_edge_buffer",
            "full_guard_compatible_source_action_required": True,
            "missing_nonfinite_or_below_threshold_behavior": (
                "fail_closed_to_no_trade"
            ),
            "recovery_source_action_mutated": False,
            "score_field": "point_selected_predicted_return",
            "threshold_grid_search_performed": False,
            "veto_action": "NO_TRADE",
            "veto_side": "NONE",
        },
        "development": profile.get("development_contract")
        == {
            "baseline_declared_position_size": DECLARED_POSITION_SIZE,
            "candidate_declared_position_size": DECLARED_POSITION_SIZE,
            "comparison_scope": "all_195_markets_in_frozen_chronological_order",
            "cost_model_changed": False,
            "declared_sizing_role": "report_only",
            "historical_development_only": True,
            "no_trade_after_cost_pnl": 0.0,
            "position_lifecycle_changed": False,
            "promotion_evidence_eligible": False,
            "statistical_gate_position_size": 1.0,
        },
        "safety": profile.get("safety") == SAFE_FALSES,
    }
    _raise_failed_checks("fixed-edge support-recovery profile", checks)


def materialize_fixed_edge_support_recovery_decisions(
    *,
    base_guard_rows: Sequence[Mapping[str, Any]],
    frozen_market_ids: Sequence[str],
    profile: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Freeze candidate decisions without reading any outcome/PnL artifact."""

    validate_fixed_edge_support_recovery_profile(profile)
    _validate_exact_order(
        base_guard_rows,
        frozen_market_ids=frozen_market_ids,
        label="base guard rows",
    )
    decisions = []
    for guard in base_guard_rows:
        _validate_base_decision(guard)
        if not _float_equal(
            guard.get("fixed_edge_buffer"),
            FIXED_EDGE_THRESHOLD,
        ):
            raise ChallengeFixedEdgeSupportRecoveryError(
                "base guard fixed edge buffer does not match preregistration"
            )
        market_id = str(guard["market_id"])
        baseline_action = str(
            guard.get("v6_7_baseline_action")
            or guard.get("baseline_action")
            or ""
        )
        if not baseline_action or baseline_action == "NO_TRADE":
            decisions.append(_no_action_decision(guard))
            continue
        baseline_side = str(guard.get("baseline_side") or "")
        if baseline_side != _side_for_action(baseline_action):
            raise ChallengeFixedEdgeSupportRecoveryError(
                "v6.7 baseline action/side does not reconcile"
            )
        decision_ts = _integer(
            guard.get("baseline_decision_ts"),
            field="baseline decision_ts",
        )
        max_input_ts = _integer(
            guard.get("baseline_max_input_ts"),
            field="baseline max_input_ts",
        )
        if max_input_ts > decision_ts:
            raise ChallengeFixedEdgeSupportRecoveryError(
                "v6.7 baseline input follows decision"
            )
        score = _optional_finite_float(
            guard.get("point_selected_predicted_return")
        )
        passed = score is not None and score >= FIXED_EDGE_THRESHOLD
        selected_action = baseline_action if passed else "NO_TRADE"
        selected_side = baseline_side if passed else "NONE"
        decisions.append(
            _decision(
                guard=guard,
                decision_ts=decision_ts,
                max_input_ts=max_input_ts,
                baseline_action=baseline_action,
                baseline_side=baseline_side,
                selected_action=selected_action,
                selected_side=selected_side,
                score=score,
                threshold_passed=passed,
            )
        )
        if str(decisions[-1]["market_id"]) != market_id:
            raise ChallengeFixedEdgeSupportRecoveryError(
                "candidate decision market identity changed"
            )
    return decisions


def build_fixed_edge_support_recovery_comparison(
    *,
    candidate_decisions: Sequence[Mapping[str, Any]],
    base_comparison_rows: Sequence[Mapping[str, Any]],
    frozen_market_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Join frozen decisions to the registered v6.7 development outcomes."""

    _validate_exact_order(
        candidate_decisions,
        frozen_market_ids=frozen_market_ids,
        label="candidate decisions",
    )
    _validate_exact_order(
        base_comparison_rows,
        frozen_market_ids=frozen_market_ids,
        label="base comparison rows",
    )
    rows = []
    for decision, base in zip(
        candidate_decisions,
        base_comparison_rows,
        strict=True,
    ):
        baseline_action = str(base.get("v6_7_action") or "")
        baseline_side = _side_for_action(baseline_action)
        if not baseline_side:
            raise ChallengeFixedEdgeSupportRecoveryError(
                "v6.7 comparison action is invalid"
            )
        baseline_pnl = _finite_float(
            base.get("v6_7_after_cost_pnl"),
            field="v6.7 baseline PnL",
        )
        action = str(decision["selected_action"])
        if action != "NO_TRADE" and (
            action != baseline_action
            or decision["selected_side"] != baseline_side
        ):
            raise ChallengeFixedEdgeSupportRecoveryError(
                "recovered action differs from frozen v6.7 baseline"
            )
        candidate_pnl = baseline_pnl if action != "NO_TRADE" else 0.0
        candidate_unit_pnl = candidate_pnl / DECLARED_POSITION_SIZE
        baseline_unit_pnl = baseline_pnl / DECLARED_POSITION_SIZE
        rows.append(
            {
                "market_id": str(decision["market_id"]),
                "candidate_id": CANDIDATE_ID,
                "candidate_action": action,
                "candidate_side": decision["selected_side"],
                "candidate_after_cost_pnl": candidate_pnl,
                "candidate_declared_position_size": DECLARED_POSITION_SIZE,
                "candidate_unit_after_cost_pnl": candidate_unit_pnl,
                "baseline_action": baseline_action,
                "baseline_side": baseline_side,
                "baseline_after_cost_pnl": baseline_pnl,
                "baseline_declared_position_size": DECLARED_POSITION_SIZE,
                "baseline_unit_after_cost_pnl": baseline_unit_pnl,
                "candidate_minus_baseline_pnl": candidate_pnl - baseline_pnl,
                "candidate_minus_baseline_unit_pnl": (
                    candidate_unit_pnl - baseline_unit_pnl
                ),
                "source_candidate_decision_id": decision["decision_id"],
                "historical_development_only": True,
                "promotion_evidence_eligible": False,
                "safety": SAFE_FALSES,
            }
        )
    return rows


def _decision(
    *,
    guard: Mapping[str, Any],
    decision_ts: int,
    max_input_ts: int,
    baseline_action: str,
    baseline_side: str,
    selected_action: str,
    selected_side: str,
    score: float | None,
    threshold_passed: bool,
) -> dict[str, Any]:
    decision = {
        "schema_version": CANDIDATE_DECISION_SCHEMA_VERSION,
        "candidate_id": CANDIDATE_ID,
        "base_candidate_id": BASE_CANDIDATE_ID,
        "market_id": str(guard["market_id"]),
        "decision_ts": decision_ts,
        "market_close_ts": _base_market_close_ts(guard),
        "max_input_ts": max_input_ts,
        "base_v8_1_selected_action": guard.get("selected_action"),
        "base_v8_1_selected_side": guard.get("selected_side"),
        "v6_7_baseline_action": baseline_action,
        "v6_7_baseline_side": baseline_side,
        "selected_action": selected_action,
        "selected_side": selected_side,
        "trade_selected": selected_action != "NO_TRADE",
        "point_selected_predicted_return": score,
        "fixed_edge_threshold_inclusive": FIXED_EDGE_THRESHOLD,
        "fixed_edge_threshold_passed": threshold_passed,
        "selection_source": (
            "v6_7_fixed_edge_support_recovery"
            if threshold_passed
            else "fixed_edge_veto"
        ),
        "selection_reason": (
            "point_selected_predicted_return_missing_or_nonfinite"
            if score is None
            else (
                "point_selected_predicted_return_at_or_above_0_025"
                if threshold_passed
                else "point_selected_predicted_return_below_0_025"
            )
        ),
        "source_guard_replay_row_id": guard.get("guard_replay_row_id"),
        "base_controller_state_before_id": guard.get(
            "controller_state_before_id"
        ),
        "base_controller_state_after_id": guard.get(
            "controller_state_after_id"
        ),
        "base_controller_state_transition_changed": False,
        "outcome_or_pnl_field_used_at_inference": False,
        "target_used_as_decision_time_input": False,
        "historical_development_only": True,
        "promotion_evidence_eligible": False,
        "safety": SAFE_FALSES,
    }
    decision["decision_id"] = canonical_json_sha256(decision)
    return decision


def _no_action_decision(guard: Mapping[str, Any]) -> dict[str, Any]:
    if (
        guard.get("selected_action") != "NO_TRADE"
        or "v6_7_no_positive_guard_compatible_action"
        not in (guard.get("selection_reason_codes") or [])
    ):
        raise ChallengeFixedEdgeSupportRecoveryError(
            "missing v6.7 action lacks fail-closed reason"
        )
    decision = {
        "schema_version": CANDIDATE_DECISION_SCHEMA_VERSION,
        "candidate_id": CANDIDATE_ID,
        "base_candidate_id": BASE_CANDIDATE_ID,
        "market_id": str(guard["market_id"]),
        "decision_ts": 0,
        "market_close_ts": None,
        "max_input_ts": None,
        "base_v8_1_selected_action": "NO_TRADE",
        "base_v8_1_selected_side": "NONE",
        "v6_7_baseline_action": None,
        "v6_7_baseline_side": None,
        "selected_action": "NO_TRADE",
        "selected_side": "NONE",
        "trade_selected": False,
        "point_selected_predicted_return": None,
        "fixed_edge_threshold_inclusive": FIXED_EDGE_THRESHOLD,
        "fixed_edge_threshold_passed": False,
        "selection_source": "v6_7_no_positive_guard_compatible_action",
        "selection_reason": "base_v6_7_no_action",
        "source_guard_replay_row_id": guard.get("guard_replay_row_id"),
        "base_controller_state_before_id": guard.get(
            "controller_state_before_id"
        ),
        "base_controller_state_after_id": guard.get(
            "controller_state_after_id"
        ),
        "base_controller_state_transition_changed": False,
        "outcome_or_pnl_field_used_at_inference": False,
        "target_used_as_decision_time_input": False,
        "historical_development_only": True,
        "promotion_evidence_eligible": False,
        "safety": SAFE_FALSES,
    }
    decision["decision_id"] = canonical_json_sha256(decision)
    return decision


def _validate_exact_order(
    rows: Sequence[Mapping[str, Any]],
    *,
    frozen_market_ids: Sequence[str],
    label: str,
) -> None:
    actual = [str(row.get("market_id") or "") for row in rows]
    if actual != list(frozen_market_ids) or len(set(actual)) != len(actual):
        raise ChallengeFixedEdgeSupportRecoveryError(
            f"{label} do not match the frozen chronological market sequence"
        )


def _finite_float(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ChallengeFixedEdgeSupportRecoveryError(
            f"{field} is not numeric"
        ) from error
    if not math.isfinite(number):
        raise ChallengeFixedEdgeSupportRecoveryError(
            f"{field} is not finite"
        )
    return number


def _optional_finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _float_equal(left: Any, right: Any, *, tolerance: float = 1e-12) -> bool:
    try:
        return math.isclose(
            float(left),
            float(right),
            rel_tol=0.0,
            abs_tol=tolerance,
        )
    except (TypeError, ValueError):
        return False


def _raise_failed_checks(label: str, checks: Mapping[str, bool]) -> None:
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ChallengeFixedEdgeSupportRecoveryError(
            f"{label} invalid: {','.join(blockers)}"
        )


__all__ = [
    "CANDIDATE_DECISION_SCHEMA_VERSION",
    "CANDIDATE_ID",
    "ChallengeFixedEdgeSupportRecoveryError",
    "DECLARED_POSITION_SIZE",
    "FIXED_EDGE_THRESHOLD",
    "PROFILE_SCHEMA_VERSION",
    "build_fixed_edge_support_recovery_comparison",
    "materialize_fixed_edge_support_recovery_decisions",
    "validate_fixed_edge_support_recovery_profile",
]
