"""Preregistered iteration-1 entry-price filter for the v8.1 challenge model."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from bigan.v8.polymarket.challenge_historical_development import SAFE_FALSES
from bigan.v8.polymarket.contracts import canonical_json_sha256

PROFILE_SCHEMA_VERSION = "bigan-v8-challenge-v8-1-entry-price-floor-profile-v1"
CANDIDATE_DECISION_SCHEMA_VERSION = (
    "bigan-v8-challenge-v8-1-entry-price-floor-decision-v1"
)
CANDIDATE_ID = "v8_1_entry_price_floor_0_30"
BASE_CANDIDATE_ID = "v8_1_primary_no_fallback"
BASE_CANDIDATE_NAME = "adaptive_support_controller_v8_1"
ENTRY_PRICE_FLOOR = 0.30

EXPECTED_LINEAGE = {
    "base_v8_1_guard_replay_sha256": (
        "7534b07c41d1a1f2afe3519e7eef9146c9b84bdda72f88a22b9d0351baa9ce2d"
    ),
    "base_v8_1_market_comparison_sha256": (
        "fce95987a10b160d7a7e6cdfd3842cc3e3b34dd138eb27093fb9f86b0a790eae"
    ),
    "base_v8_1_runtime_targets_sha256": (
        "f589421b7b483b9cfc42a486ddc484743f4114250f7c3c363586786a836fcb3e"
    ),
    "exact_195_five_action_rows_sha256": (
        "134425d9f38ffdebbf72043a8b802e95bbb87eebbe7e8d39bfa5d6f8b98828f7"
    ),
    "exact_195_market_ids_sha256": (
        "fef9eda7b8dac138b88c75f96b010bd40953795b2bcf7424debf77a004e06883"
    ),
    "preregistration_commit": "2f8c305cb4133b5a3c69c7500877889e0cdcd2c2",
    "preregistration_sha256": (
        "883a60ec4acfe02b9ae5216bd582d1579622c249aa6ffa7845c3e28a32d2578e"
    ),
}


class ChallengeEntryPriceFloorError(ValueError):
    """Raised when the preregistered price-floor candidate cannot fail closed."""


def validate_entry_price_floor_profile(profile: Mapping[str, Any]) -> None:
    """Validate the exact preregistered iteration-1 policy."""

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
            "price_floor_source": (
                "existing_v7_paper_runtime_min_entry_price_"
                "not_exact_195_grid_search"
            ),
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
    _raise_failed_checks("entry-price-floor profile", checks)


def apply_entry_price_floor(
    *,
    base_decision: Mapping[str, Any],
    matched_action_row: Mapping[str, Any] | None,
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the terminal price filter using decision-time data only."""

    validate_entry_price_floor_profile(profile)
    return _apply_entry_price_floor_validated(
        base_decision=base_decision,
        matched_action_row=matched_action_row,
    )


def materialize_entry_price_floor_decisions(
    *,
    base_guard_rows: Sequence[Mapping[str, Any]],
    five_action_rows: Sequence[Mapping[str, Any]],
    frozen_market_ids: Sequence[str],
    profile: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Materialize the candidate before any outcome/PnL artifact is read."""

    validate_entry_price_floor_profile(profile)
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
        if not all((key[0], key[2])) or key in action_index:
            raise ChallengeEntryPriceFloorError(
                "five-action decision identity is missing or duplicated"
            )
        action_index[key] = row

    decisions = []
    for base_decision in base_guard_rows:
        action = str(base_decision.get("selected_action") or "")
        action_row = None
        if action != "NO_TRADE":
            key = (
                str(base_decision.get("market_id") or ""),
                _integer(
                    base_decision.get("decision_ts"),
                    field="base decision_ts",
                ),
                action,
            )
            action_row = action_index.get(key)
            if action_row is None:
                raise ChallengeEntryPriceFloorError(
                    "selected trade has no exact outcome-blind action row"
                )
        decisions.append(
            _apply_entry_price_floor_validated(
                base_decision=base_decision,
                matched_action_row=action_row,
            )
        )
    return decisions


def build_entry_price_floor_comparison(
    *,
    candidate_decisions: Sequence[Mapping[str, Any]],
    base_comparison_rows: Sequence[Mapping[str, Any]],
    base_runtime_targets: Sequence[Mapping[str, Any]],
    frozen_market_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Join frozen candidate decisions to already-opened development outcomes."""

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
    targets_by_market: dict[str, Mapping[str, Any]] = {}
    for target in base_runtime_targets:
        market_id = str(target.get("market_id") or "")
        if not market_id or market_id in targets_by_market:
            raise ChallengeEntryPriceFloorError(
                "runtime target market identity is missing or duplicated"
            )
        _validate_runtime_target(target)
        targets_by_market[market_id] = target

    comparison_rows = []
    for decision, base_row in zip(
        candidate_decisions,
        base_comparison_rows,
        strict=True,
    ):
        market_id = str(decision["market_id"])
        base_action = str(decision["base_selected_action"])
        candidate_action = str(decision["selected_action"])
        target = targets_by_market.get(market_id)
        base_candidate_pnl = _finite_float(
            base_row.get("challenge_after_cost_pnl"),
            field="base challenge PnL",
        )
        if base_action == "NO_TRADE":
            if target is not None or not _float_equal(base_candidate_pnl, 0.0):
                raise ChallengeEntryPriceFloorError(
                    "base NO_TRADE target/PnL does not reconcile"
                )
        else:
            if target is None:
                raise ChallengeEntryPriceFloorError(
                    "base accepted trade has no runtime target"
                )
            if (
                target.get("action") != base_action
                or target.get("side") != decision["base_selected_side"]
                or not _float_equal(
                    target.get(
                        "runtime_policy_after_cost_net_pnl_at_frozen_size"
                    ),
                    base_candidate_pnl,
                )
            ):
                raise ChallengeEntryPriceFloorError(
                    "base accepted trade target does not reconcile"
                )

        candidate_pnl = (
            base_candidate_pnl if candidate_action != "NO_TRADE" else 0.0
        )
        baseline_pnl = _finite_float(
            base_row.get("v6_7_after_cost_pnl"),
            field="v6.7 baseline PnL",
        )
        baseline_action = str(base_row.get("v6_7_action") or "")
        baseline_side = _side_for_action(baseline_action)
        if not baseline_side:
            raise ChallengeEntryPriceFloorError(
                "v6.7 baseline action is invalid"
            )
        comparison_rows.append(
            {
                "market_id": market_id,
                "candidate_id": CANDIDATE_ID,
                "candidate_action": candidate_action,
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
    if set(targets_by_market) != {
        str(row["market_id"])
        for row in candidate_decisions
        if row["base_selected_action"] != "NO_TRADE"
    }:
        raise ChallengeEntryPriceFloorError(
            "runtime targets do not exactly cover base accepted trades"
        )
    return comparison_rows


def _apply_entry_price_floor_validated(
    *,
    base_decision: Mapping[str, Any],
    matched_action_row: Mapping[str, Any] | None,
) -> dict[str, Any]:
    _validate_base_decision(base_decision)
    market_id = str(base_decision["market_id"])
    decision_ts = _integer(
        base_decision.get("decision_ts"),
        field="base decision_ts",
    )
    base_action = str(base_decision["selected_action"])
    base_side = str(base_decision["selected_side"])
    entry_price: float | None = None
    action_row_sha256: str | None = None
    filter_evaluated = base_action != "NO_TRADE"
    filter_passed = False
    if filter_evaluated:
        if matched_action_row is None:
            raise ChallengeEntryPriceFloorError(
                "trade decision requires its matched action row"
            )
        entry_price = _validated_entry_price(
            matched_action_row,
            market_id=market_id,
            decision_ts=decision_ts,
            action=base_action,
        )
        action_row_sha256 = str(matched_action_row["action_row_sha256"])
        filter_passed = entry_price >= ENTRY_PRICE_FLOOR
    elif matched_action_row is not None:
        raise ChallengeEntryPriceFloorError(
            "base NO_TRADE must not consume an action row"
        )

    if filter_passed:
        selected_action = base_action
        selected_side = base_side
        reason = "entry_price_at_or_above_0_30_floor"
    elif filter_evaluated:
        selected_action = "NO_TRADE"
        selected_side = "NONE"
        reason = "entry_price_below_0_30_floor"
    else:
        selected_action = "NO_TRADE"
        selected_side = "NONE"
        reason = "base_v8_1_no_trade"

    decision = {
        "schema_version": CANDIDATE_DECISION_SCHEMA_VERSION,
        "candidate_id": CANDIDATE_ID,
        "base_candidate_id": BASE_CANDIDATE_ID,
        "market_id": market_id,
        "decision_ts": decision_ts,
        "market_close_ts": _integer(
            base_decision.get("market_close_ts"),
            field="market_close_ts",
        ),
        "max_input_ts": _integer(
            _base_max_input_ts(base_decision),
            field="max_input_ts",
        ),
        "base_selected_action": base_action,
        "base_selected_side": base_side,
        "selected_action": selected_action,
        "selected_side": selected_side,
        "trade_selected": selected_action != "NO_TRADE",
        "entry_price": entry_price,
        "entry_price_floor_inclusive": ENTRY_PRICE_FLOOR,
        "entry_price_filter_evaluated": filter_evaluated,
        "entry_price_filter_passed": filter_passed,
        "selection_reason": reason,
        "source_action_row_sha256": action_row_sha256,
        "source_guard_replay_row_id": base_decision.get(
            "guard_replay_row_id"
        ),
        "base_controller_state_before_id": base_decision.get(
            "controller_state_before_id"
        ),
        "base_controller_state_after_id": base_decision.get(
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


def _validate_base_decision(base_decision: Mapping[str, Any]) -> None:
    action = str(base_decision.get("selected_action") or "")
    side = str(base_decision.get("selected_side") or "")
    checks = {
        "identity": bool(base_decision.get("market_id"))
        and base_decision.get("candidate_name") == BASE_CANDIDATE_NAME,
        "action": side == _side_for_action(action),
        "guard": (
            action == "NO_TRADE"
            or base_decision.get("execution_guard_order_allowed") is True
        ),
        "causal": (
            base_decision.get("target_used_as_decision_time_input") is False
            or (
                action == "NO_TRADE"
                and base_decision.get("target_used_as_decision_time_input")
                is None
                and "v6_7_no_positive_guard_compatible_action"
                in (base_decision.get("selection_reason_codes") or [])
            )
        )
        and base_decision.get("outcome_or_pnl_field_used_at_inference")
        is False
        and base_decision.get("labels_outcomes_or_pnl_opened") is False,
        "immutable_score": base_decision.get("source_score_mutated") is False,
        "safety": base_decision.get("capital_at_risk") is False
        and base_decision.get("polymarket_write_enabled") is False
        and base_decision.get("wallet_signing_enabled") is False
        and base_decision.get("v8_execution_handoff_allowed") is False
        and base_decision.get("promotion_evidence_eligible") is False
        and base_decision.get("live_trading_enabled") is False,
    }
    _raise_failed_checks("base v8.1 decision", checks)
    decision_ts = _integer(
        base_decision.get("decision_ts"),
        field="base decision_ts",
    )
    if _base_max_input_ts(base_decision) > decision_ts:
        raise ChallengeEntryPriceFloorError(
            "base decision uses input after decision_ts"
        )


def _base_max_input_ts(base_decision: Mapping[str, Any]) -> int:
    direct = base_decision.get("max_input_ts")
    if direct is not None:
        return _integer(direct, field="base max_input_ts")
    timestamps = [
        _integer(value, field=f"base {field}")
        for field in ("baseline_max_input_ts", "opposite_max_input_ts")
        if (value := base_decision.get(field)) is not None
    ]
    if not timestamps:
        raise ChallengeEntryPriceFloorError(
            "base decision has no causal max_input_ts"
        )
    return max(timestamps)


def _validated_entry_price(
    action_row: Mapping[str, Any],
    *,
    market_id: str,
    decision_ts: int,
    action: str,
) -> float:
    supplied_sha256 = str(action_row.get("action_row_sha256") or "")
    actual_sha256 = canonical_json_sha256(
        {
            key: value
            for key, value in action_row.items()
            if key != "action_row_sha256"
        }
    )
    features = dict(action_row.get("decision_time_features") or {})
    snapshot = dict(action_row.get("microstructure_snapshot") or {})
    entry_price = _finite_float(
        features.get("execution_price"),
        field="decision-time execution price",
    )
    checks = {
        "identity": action_row.get("market_id") == market_id
        and action_row.get("decision_ts") == decision_ts
        and action_row.get("action") == action,
        "hash": supplied_sha256 == actual_sha256,
        "causal": action_row.get("target_used_as_decision_input") is False
        and action_row.get("outcome_fields_used_as_decision_input") is False,
        "time": _integer(
            action_row.get("max_input_ts"),
            field="action max_input_ts",
        )
        <= decision_ts,
        "price": 0.0 <= entry_price <= 1.0
        and _float_equal(snapshot.get("entry_ask"), entry_price),
        "safety": action_row.get("paper_only") is True
        and action_row.get("capital_at_risk") is False,
    }
    _raise_failed_checks("matched decision-time action row", checks)
    return entry_price


def _validate_runtime_target(target: Mapping[str, Any]) -> None:
    checks = {
        "schema": target.get("schema_version")
        == "bigan-v8-runtime-aligned-sbc-net-return-v6-4-target-row-v1",
        "target_timing": target.get("target_used_as_decision_time_input")
        is False
        and target.get("target_available_only_post_exit_or_official_resolution")
        is True,
        "size": _float_equal(target.get("paper_position_size"), 0.2),
        "cost": target.get("cost_fields_subtracted_exactly_once") is True,
        "safety": target.get("capital_at_risk") is False
        and target.get("polymarket_write_enabled") is False
        and target.get("wallet_signing_enabled") is False
        and target.get("v8_execution_handoff_allowed") is False
        and target.get("promotion_evidence_eligible") is False,
    }
    _raise_failed_checks("base runtime target", checks)


def _validate_exact_order(
    rows: Sequence[Mapping[str, Any]],
    *,
    frozen_market_ids: Sequence[str],
    label: str,
) -> None:
    actual = [str(row.get("market_id") or "") for row in rows]
    if actual != list(frozen_market_ids) or len(set(actual)) != len(actual):
        raise ChallengeEntryPriceFloorError(
            f"{label} do not match the frozen chronological market sequence"
        )


def _side_for_action(action: str) -> str:
    if action == "NO_TRADE":
        return "NONE"
    if action.startswith("BUY_UP_"):
        return "UP"
    if action.startswith("BUY_DOWN_"):
        return "DOWN"
    return ""


def _finite_float(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ChallengeEntryPriceFloorError(f"{field} is not numeric") from error
    if not math.isfinite(number):
        raise ChallengeEntryPriceFloorError(f"{field} is not finite")
    return number


def _integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ChallengeEntryPriceFloorError(f"{field} is not an integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ChallengeEntryPriceFloorError(
            f"{field} is not an integer"
        ) from error
    if number < 0 or value != number:
        raise ChallengeEntryPriceFloorError(f"{field} is not a nonnegative integer")
    return number


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
        raise ChallengeEntryPriceFloorError(
            f"{label} invalid: {','.join(blockers)}"
        )


__all__ = [
    "BASE_CANDIDATE_ID",
    "CANDIDATE_DECISION_SCHEMA_VERSION",
    "CANDIDATE_ID",
    "ChallengeEntryPriceFloorError",
    "ENTRY_PRICE_FLOOR",
    "PROFILE_SCHEMA_VERSION",
    "apply_entry_price_floor",
    "build_entry_price_floor_comparison",
    "materialize_entry_price_floor_decisions",
    "validate_entry_price_floor_profile",
]
