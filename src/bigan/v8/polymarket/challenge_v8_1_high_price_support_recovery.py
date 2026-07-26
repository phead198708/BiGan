"""Preregistered iteration-2 high-price support recovery for v8.1."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from bigan.v8.polymarket.challenge_historical_development import SAFE_FALSES
from bigan.v8.polymarket.challenge_v8_1_entry_price_floor import (
    BASE_CANDIDATE_ID,
    ENTRY_PRICE_FLOOR,
    ChallengeEntryPriceFloorError,
    _base_market_close_ts,
    _integer,
    _side_for_action,
    _validate_base_decision,
    _validated_entry_price,
)
from bigan.v8.polymarket.contracts import canonical_json_sha256

PROFILE_SCHEMA_VERSION = (
    "bigan-v8-challenge-v8-1-high-price-support-recovery-profile-v1"
)
CANDIDATE_DECISION_SCHEMA_VERSION = (
    "bigan-v8-challenge-v8-1-high-price-support-recovery-decision-v1"
)
CANDIDATE_ID = "v8_1_high_price_support_recovery_0_30"
EXPECTED_LINEAGE = {
    "base_v8_1_guard_replay_sha256": (
        "7534b07c41d1a1f2afe3519e7eef9146c9b84bdda72f88a22b9d0351baa9ce2d"
    ),
    "base_v8_1_market_comparison_sha256": (
        "fce95987a10b160d7a7e6cdfd3842cc3e3b34dd138eb27093fb9f86b0a790eae"
    ),
    "exact_195_five_action_rows_sha256": (
        "134425d9f38ffdebbf72043a8b802e95bbb87eebbe7e8d39bfa5d6f8b98828f7"
    ),
    "exact_195_market_ids_sha256": (
        "fef9eda7b8dac138b88c75f96b010bd40953795b2bcf7424debf77a004e06883"
    ),
    "iteration_001_entry_sha256": (
        "76b1ea851c40c3b66aa47e762cfcfdeaada0b6372232a1f4c4365cb6cbebff95"
    ),
    "iteration_001_result_sha256": (
        "80baab4220f48d2cfb98d4ccd25a9b4807462691cfc13f0a40fb96f381a90782"
    ),
    "preregistration_commit": "42328b2743ebf54684007082d255ab16a5160ffa",
    "preregistration_sha256": (
        "e774af3e31fc7b932561745aee330bef18a21e666d8e7e54d4f71be8e550c1f6"
    ),
}


class ChallengeHighPriceSupportRecoveryError(ChallengeEntryPriceFloorError):
    """Raised when high-price support recovery cannot fail closed."""


def validate_high_price_support_recovery_profile(
    profile: Mapping[str, Any],
) -> None:
    """Validate the exact iteration-2 policy."""

    checks = {
        "identity": profile.get("schema_version") == PROFILE_SCHEMA_VERSION
        and profile.get("candidate_id") == CANDIDATE_ID,
        "lineage": profile.get("lineage") == EXPECTED_LINEAGE,
        "policy": profile.get("policy")
        == {
            "base_candidate_id": BASE_CANDIDATE_ID,
            "base_controller_state_transition_changed": False,
            "entry_price_floor_inclusive": ENTRY_PRICE_FLOOR,
            "entry_price_source": (
                "matched_outcome_blind_five_action_row."
                "decision_time_features.execution_price"
            ),
            "filter_application_order": (
                "after_v8_1_controller_and_full_guard_before_final_trade"
            ),
            "missing_or_nonfinite_entry_price_behavior": (
                "fail_closed_to_no_trade"
            ),
            "primary_path": (
                "keep_v8_1_trade_only_when_execution_price_at_least_0_30"
            ),
            "recovery_path": (
                "recover_existing_v6_7_guard_compatible_action_only_when_"
                "execution_price_at_least_0_30"
            ),
            "recovery_source_action_mutated": False,
            "threshold_grid_search_performed": False,
            "veto_action": "NO_TRADE",
            "veto_side": "NONE",
        },
        "development": profile.get("development_contract")
        == {
            "comparison_scope": (
                "all_195_markets_in_frozen_chronological_order"
            ),
            "cost_model_changed": False,
            "historical_development_only": True,
            "no_trade_after_cost_pnl": 0.0,
            "position_lifecycle_changed": False,
            "position_size": 0.2,
            "promotion_evidence_eligible": False,
        },
        "safety": profile.get("safety") == SAFE_FALSES,
    }
    _raise_failed_checks("high-price support-recovery profile", checks)


def materialize_high_price_support_recovery_decisions(
    *,
    base_guard_rows: Sequence[Mapping[str, Any]],
    five_action_rows: Sequence[Mapping[str, Any]],
    frozen_market_ids: Sequence[str],
    profile: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Materialize the recovery candidate without outcome/PnL inputs."""

    validate_high_price_support_recovery_profile(profile)
    _validate_exact_order(
        base_guard_rows,
        frozen_market_ids=frozen_market_ids,
        label="base guard rows",
    )
    action_index: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for row in five_action_rows:
        key = (
            str(row.get("market_id") or ""),
            _integer(row.get("decision_ts"), field="action decision_ts"),
            str(row.get("action") or ""),
        )
        if not key[0] or not key[2] or key in action_index:
            raise ChallengeHighPriceSupportRecoveryError(
                "five-action decision identity is missing or duplicated"
            )
        action_index[key] = row

    decisions = []
    for guard in base_guard_rows:
        _validate_base_decision(guard)
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
            raise ChallengeHighPriceSupportRecoveryError(
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
            raise ChallengeHighPriceSupportRecoveryError(
                "v6.7 baseline input follows decision"
            )
        action_row = action_index.get(
            (market_id, decision_ts, baseline_action)
        )
        if action_row is None:
            raise ChallengeHighPriceSupportRecoveryError(
                "v6.7 recovery action has no exact outcome-blind row"
            )
        entry_price = _validated_entry_price(
            action_row,
            market_id=market_id,
            decision_ts=decision_ts,
            action=baseline_action,
        )
        passed = entry_price >= ENTRY_PRICE_FLOOR
        base_v8_action = str(guard.get("selected_action") or "")
        if passed:
            selected_action = baseline_action
            selected_side = baseline_side
            source = (
                "v8_1_primary"
                if base_v8_action != "NO_TRADE"
                else "v6_7_high_price_support_recovery"
            )
            reason = "entry_price_at_or_above_0_30_floor"
        else:
            selected_action = "NO_TRADE"
            selected_side = "NONE"
            source = "price_floor_veto"
            reason = "entry_price_below_0_30_floor"
        decisions.append(
            _decision(
                guard=guard,
                decision_ts=decision_ts,
                max_input_ts=max_input_ts,
                baseline_action=baseline_action,
                baseline_side=baseline_side,
                selected_action=selected_action,
                selected_side=selected_side,
                entry_price=entry_price,
                filter_passed=passed,
                selection_source=source,
                selection_reason=reason,
                action_row_sha256=str(action_row["action_row_sha256"]),
            )
        )
    return decisions


def build_high_price_support_recovery_comparison(
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
            raise ChallengeHighPriceSupportRecoveryError(
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
            raise ChallengeHighPriceSupportRecoveryError(
                "recovered action differs from frozen v6.7 baseline"
            )
        candidate_pnl = baseline_pnl if action != "NO_TRADE" else 0.0
        rows.append(
            {
                "market_id": str(decision["market_id"]),
                "candidate_id": CANDIDATE_ID,
                "candidate_action": action,
                "candidate_side": decision["selected_side"],
                "candidate_after_cost_pnl": candidate_pnl,
                "baseline_action": baseline_action,
                "baseline_side": baseline_side,
                "baseline_after_cost_pnl": baseline_pnl,
                "candidate_minus_baseline_pnl": (
                    candidate_pnl - baseline_pnl
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
    entry_price: float,
    filter_passed: bool,
    selection_source: str,
    selection_reason: str,
    action_row_sha256: str,
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
        "selection_source": selection_source,
        "selection_reason": selection_reason,
        "entry_price": entry_price,
        "entry_price_floor_inclusive": ENTRY_PRICE_FLOOR,
        "entry_price_filter_passed": filter_passed,
        "source_action_row_sha256": action_row_sha256,
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
        raise ChallengeHighPriceSupportRecoveryError(
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
        "selection_source": "v6_7_no_positive_guard_compatible_action",
        "selection_reason": "base_v6_7_no_action",
        "entry_price": None,
        "entry_price_floor_inclusive": ENTRY_PRICE_FLOOR,
        "entry_price_filter_passed": False,
        "source_action_row_sha256": None,
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
        raise ChallengeHighPriceSupportRecoveryError(
            f"{label} do not match the frozen chronological market sequence"
        )


def _finite_float(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ChallengeHighPriceSupportRecoveryError(
            f"{field} is not numeric"
        ) from error
    if not math.isfinite(number):
        raise ChallengeHighPriceSupportRecoveryError(
            f"{field} is not finite"
        )
    return number


def _raise_failed_checks(label: str, checks: Mapping[str, bool]) -> None:
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ChallengeHighPriceSupportRecoveryError(
            f"{label} invalid: {','.join(blockers)}"
        )


__all__ = [
    "CANDIDATE_DECISION_SCHEMA_VERSION",
    "CANDIDATE_ID",
    "ChallengeHighPriceSupportRecoveryError",
    "PROFILE_SCHEMA_VERSION",
    "build_high_price_support_recovery_comparison",
    "materialize_high_price_support_recovery_decisions",
    "validate_high_price_support_recovery_profile",
]
