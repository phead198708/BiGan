"""Preregistered slot-5 fixed-edge plus price-floor candidate."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from bigan.v8.polymarket.challenge_historical_development import SAFE_FALSES
from bigan.v8.polymarket.challenge_v8_1_entry_price_floor import (
    _integer,
    _side_for_action,
    _validated_entry_price,
)
from bigan.v8.polymarket.challenge_v8_1_fixed_edge_support_recovery import (
    CANDIDATE_ID as FIXED_EDGE_CANDIDATE_ID,
)
from bigan.v8.polymarket.challenge_v8_1_fixed_edge_support_recovery import (
    DECLARED_POSITION_SIZE,
    FIXED_EDGE_THRESHOLD,
    materialize_fixed_edge_support_recovery_decisions,
    validate_fixed_edge_support_recovery_profile,
)
from bigan.v8.polymarket.contracts import canonical_json_sha256

PROFILE_SCHEMA_VERSION = (
    "bigan-v8-challenge-v8-1-fixed-edge-price-floor-profile-v1"
)
CANDIDATE_DECISION_SCHEMA_VERSION = (
    "bigan-v8-challenge-v8-1-fixed-edge-price-floor-decision-v1"
)
CANDIDATE_ID = "v8_1_fixed_edge_0_025_price_floor_0_30"
ENTRY_PRICE_FLOOR = 0.30
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
    "iteration_004_entry_semantic_sha256": (
        "3c1ce6fcddf650ff9bd30bd46c3761b5df7fe785c88b06fa0b07ec9084a265b1"
    ),
    "preregistration_commit": "986e96f6cc3d1d4c418231c6ffd99581696ee21f",
    "preregistration_sha256": (
        "5e5d37dbc4fe8bf7fd3badec5d4ba458152faaa4c755169fa7de5cbb427d5f6c"
    ),
    "slot_004_profile_sha256": (
        "aaf1b22f6c2b372ca48122ae672fe45b019998048934e611219f088d26746ab2"
    ),
    "success_standard_v2_sha256": (
        "01b6d0c80cd9f54cf78523e556788788dd3ac6324dc1865d385f6f4cf2dcb9bb"
    ),
}


class ChallengeFixedEdgePriceFloorError(ValueError):
    """Raised when slot 5 cannot remain within its preregistration."""


def validate_fixed_edge_price_floor_profile(
    profile: Mapping[str, Any],
) -> None:
    """Validate the exact single-hypothesis slot-5 policy."""

    checks = {
        "identity": profile.get("schema_version") == PROFILE_SCHEMA_VERSION
        and profile.get("candidate_id") == CANDIDATE_ID,
        "lineage": profile.get("lineage") == EXPECTED_LINEAGE,
        "policy": profile.get("policy")
        == {
            "base_candidate_id": FIXED_EDGE_CANDIDATE_ID,
            "base_controller_state_transition_changed": False,
            "entry_price_floor_inclusive": ENTRY_PRICE_FLOOR,
            "entry_price_floor_source": (
                "existing_v7_paper_runtime_min_entry_price"
            ),
            "entry_price_source": (
                "matched_outcome_blind_five_action_row."
                "decision_time_features.execution_price"
            ),
            "filter_application_order": (
                "after_fixed_edge_0_025_and_full_guard_before_final_trade"
            ),
            "fixed_edge_threshold_inclusive": FIXED_EDGE_THRESHOLD,
            "missing_nonfinite_or_below_either_threshold_behavior": (
                "fail_closed_to_no_trade"
            ),
            "source_action_mutated": False,
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
    _raise_failed_checks("fixed-edge price-floor profile", checks)


def materialize_fixed_edge_price_floor_decisions(
    *,
    base_guard_rows: Sequence[Mapping[str, Any]],
    five_action_rows: Sequence[Mapping[str, Any]],
    frozen_market_ids: Sequence[str],
    fixed_edge_profile: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Freeze slot-5 decisions without reading outcome/PnL artifacts."""

    validate_fixed_edge_price_floor_profile(profile)
    validate_fixed_edge_support_recovery_profile(fixed_edge_profile)
    fixed_edge_decisions = materialize_fixed_edge_support_recovery_decisions(
        base_guard_rows=base_guard_rows,
        frozen_market_ids=frozen_market_ids,
        profile=fixed_edge_profile,
    )
    action_index: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for row in five_action_rows:
        key = (
            str(row.get("market_id") or ""),
            _integer(row.get("decision_ts"), field="action decision_ts"),
            str(row.get("action") or ""),
        )
        if not key[0] or not key[2] or key in action_index:
            raise ChallengeFixedEdgePriceFloorError(
                "five-action decision identity is missing or duplicated"
            )
        action_index[key] = row

    decisions = []
    for base in fixed_edge_decisions:
        action = str(base["selected_action"])
        if action == "NO_TRADE":
            decisions.append(_no_trade_decision(base))
            continue
        decision_ts = _integer(
            base.get("decision_ts"),
            field="fixed-edge decision_ts",
        )
        action_row = action_index.get(
            (str(base["market_id"]), decision_ts, action)
        )
        if action_row is None:
            raise ChallengeFixedEdgePriceFloorError(
                "fixed-edge trade has no exact outcome-blind action row"
            )
        entry_price = _validated_entry_price(
            action_row,
            market_id=str(base["market_id"]),
            decision_ts=decision_ts,
            action=action,
        )
        passed = entry_price >= ENTRY_PRICE_FLOOR
        decisions.append(
            _decision(
                base=base,
                selected_action=action if passed else "NO_TRADE",
                selected_side=str(base["selected_side"]) if passed else "NONE",
                entry_price=entry_price,
                price_passed=passed,
                action_row_sha256=str(action_row["action_row_sha256"]),
            )
        )
    _validate_exact_order(
        decisions,
        frozen_market_ids=frozen_market_ids,
        label="candidate decisions",
    )
    return decisions


def build_fixed_edge_price_floor_comparison(
    *,
    candidate_decisions: Sequence[Mapping[str, Any]],
    base_comparison_rows: Sequence[Mapping[str, Any]],
    frozen_market_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Join frozen slot-5 decisions to the registered historical outcomes."""

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
            raise ChallengeFixedEdgePriceFloorError(
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
            raise ChallengeFixedEdgePriceFloorError(
                "selected action differs from frozen v6.7 baseline"
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
    base: Mapping[str, Any],
    selected_action: str,
    selected_side: str,
    entry_price: float,
    price_passed: bool,
    action_row_sha256: str,
) -> dict[str, Any]:
    decision = {
        "schema_version": CANDIDATE_DECISION_SCHEMA_VERSION,
        "candidate_id": CANDIDATE_ID,
        "base_candidate_id": FIXED_EDGE_CANDIDATE_ID,
        "market_id": str(base["market_id"]),
        "decision_ts": base["decision_ts"],
        "market_close_ts": base["market_close_ts"],
        "max_input_ts": base["max_input_ts"],
        "v6_7_baseline_action": base["v6_7_baseline_action"],
        "v6_7_baseline_side": base["v6_7_baseline_side"],
        "selected_action": selected_action,
        "selected_side": selected_side,
        "trade_selected": selected_action != "NO_TRADE",
        "point_selected_predicted_return": base[
            "point_selected_predicted_return"
        ],
        "fixed_edge_threshold_inclusive": FIXED_EDGE_THRESHOLD,
        "fixed_edge_threshold_passed": True,
        "entry_price": entry_price,
        "entry_price_floor_inclusive": ENTRY_PRICE_FLOOR,
        "entry_price_filter_passed": price_passed,
        "selection_reason": (
            "fixed_edge_and_entry_price_floor_passed"
            if price_passed
            else "entry_price_below_0_30_floor"
        ),
        "source_fixed_edge_decision_id": base["decision_id"],
        "source_action_row_sha256": action_row_sha256,
        "base_controller_state_before_id": base[
            "base_controller_state_before_id"
        ],
        "base_controller_state_after_id": base[
            "base_controller_state_after_id"
        ],
        "base_controller_state_transition_changed": False,
        "outcome_or_pnl_field_used_at_inference": False,
        "target_used_as_decision_time_input": False,
        "historical_development_only": True,
        "promotion_evidence_eligible": False,
        "safety": SAFE_FALSES,
    }
    decision["decision_id"] = canonical_json_sha256(decision)
    return decision


def _no_trade_decision(base: Mapping[str, Any]) -> dict[str, Any]:
    decision = {
        "schema_version": CANDIDATE_DECISION_SCHEMA_VERSION,
        "candidate_id": CANDIDATE_ID,
        "base_candidate_id": FIXED_EDGE_CANDIDATE_ID,
        "market_id": str(base["market_id"]),
        "decision_ts": base["decision_ts"],
        "market_close_ts": base["market_close_ts"],
        "max_input_ts": base["max_input_ts"],
        "v6_7_baseline_action": base["v6_7_baseline_action"],
        "v6_7_baseline_side": base["v6_7_baseline_side"],
        "selected_action": "NO_TRADE",
        "selected_side": "NONE",
        "trade_selected": False,
        "point_selected_predicted_return": base[
            "point_selected_predicted_return"
        ],
        "fixed_edge_threshold_inclusive": FIXED_EDGE_THRESHOLD,
        "fixed_edge_threshold_passed": base["fixed_edge_threshold_passed"],
        "entry_price": None,
        "entry_price_floor_inclusive": ENTRY_PRICE_FLOOR,
        "entry_price_filter_passed": False,
        "selection_reason": "base_fixed_edge_candidate_no_trade",
        "source_fixed_edge_decision_id": base["decision_id"],
        "source_action_row_sha256": None,
        "base_controller_state_before_id": base[
            "base_controller_state_before_id"
        ],
        "base_controller_state_after_id": base[
            "base_controller_state_after_id"
        ],
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
        raise ChallengeFixedEdgePriceFloorError(
            f"{label} do not match the frozen chronological market sequence"
        )


def _finite_float(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ChallengeFixedEdgePriceFloorError(
            f"{field} is not numeric"
        ) from error
    if not math.isfinite(number):
        raise ChallengeFixedEdgePriceFloorError(
            f"{field} is not finite"
        )
    return number


def _raise_failed_checks(label: str, checks: Mapping[str, bool]) -> None:
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ChallengeFixedEdgePriceFloorError(
            f"{label} invalid: {','.join(blockers)}"
        )


__all__ = [
    "CANDIDATE_DECISION_SCHEMA_VERSION",
    "CANDIDATE_ID",
    "ChallengeFixedEdgePriceFloorError",
    "ENTRY_PRICE_FLOOR",
    "PROFILE_SCHEMA_VERSION",
    "build_fixed_edge_price_floor_comparison",
    "materialize_fixed_edge_price_floor_decisions",
    "validate_fixed_edge_price_floor_profile",
]
