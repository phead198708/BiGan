"""Preregistered full-window future gate for challenge attempt-002."""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from bigan.v8.polymarket.challenge_historical_development import SAFE_FALSES

ATTEMPT_002_PREREGISTRATION_SCHEMA_VERSION = (
    "bigan-v8-challenge-attempt-002-preregistration-v1"
)
ATTEMPT_002_RESULT_SCHEMA_VERSION = (
    "bigan-v8-challenge-attempt-002-future-evaluation-result-v1"
)
CANDIDATE_ID = "v8_1_entry_price_floor_0_30_sized_1_0"
BASELINE_ID = "matched_frozen_v6_7"
TRADE_ACTIONS = {
    "BUY_UP_SELL_BEFORE_CLOSE": "UP",
    "BUY_DOWN_SELL_BEFORE_CLOSE": "DOWN",
}
NO_TRADE = "NO_TRADE"


class ChallengeAttempt002Error(ValueError):
    """Raised when attempt-002 preregistration or evidence fails closed."""


def validate_attempt_002_preregistration(
    protocol: Mapping[str, Any],
    *,
    expected_lineage: Mapping[str, str] | None = None,
) -> None:
    """Validate the frozen future protocol before collection authorization."""

    lineage = dict(protocol.get("lineage") or {})
    window = dict(protocol.get("future_window") or {})
    freeze = dict(protocol.get("decision_freeze") or {})
    settlement = dict(protocol.get("settlement") or {})
    mapping = dict(protocol.get("target_mapping") or {})
    evaluation = dict(protocol.get("evaluation") or {})
    paired = dict(evaluation.get("full_window_paired_gate") or {})
    absolute = dict(evaluation.get("absolute_candidate_gate") or {})
    robustness = dict(evaluation.get("robustness_gates") or {})
    halves = dict(robustness.get("chronological_halves") or {})
    support = dict(evaluation.get("support") or {})
    concentration = dict(evaluation.get("concentration_diagnostics") or {})
    eligibility = dict(protocol.get("historical_eligibility") or {})
    alpha = dict(protocol.get("alpha_spending") or {})
    promotion = dict(protocol.get("promotion_evidence") or {})
    freeze_created_ts = protocol.get("preregistration_freeze_created_ts")
    preregistered_at = protocol.get("preregistered_at")

    checks = {
        "schema": protocol.get("schema_version")
        == ATTEMPT_002_PREREGISTRATION_SCHEMA_VERSION,
        "identity": protocol.get("attempt_id")
        == "v8-1-challenger-future-attempt-002"
        and protocol.get("issue") == 262
        and protocol.get("goal")
        == "challenge_model_promote_to_champion_model"
        and protocol.get("model_version") == "v8.1"
        and protocol.get("candidate_id") == CANDIDATE_ID
        and protocol.get("baseline_id") == BASELINE_ID,
        "freeze": protocol.get("frozen") is True
        and protocol.get("preregistered_before_collection") is True
        and _positive_integer(freeze_created_ts)
        and isinstance(preregistered_at, str)
        and preregistered_at.endswith("Z"),
        "lineage": bool(lineage)
        and all(
            _is_sha256(value) or _is_git_commit(value)
            for value in lineage.values()
        )
        and (
            expected_lineage is None
            or lineage == dict(expected_lineage)
        ),
        "historical_eligibility": eligibility
        == {
            "iteration_number": 3,
            "all_historical_success_criteria_passed": True,
            "attempt_002_preregistration_allowed": True,
            "historical_result_is_promotion_evidence": False,
            "development_iterations_consumed": 3,
            "development_iteration_limit": 5,
            "further_historical_iteration_allowed_after_success": False,
        },
        "future_window": window
        == {
            "market_family": "btc_updown_5m",
            "exact_quality_valid_market_count": 120,
            "selection_rule": (
                "chronological_earliest_quality_valid_after_freeze_boundary"
            ),
            "strictly_future_and_disjoint_from_all_development_data": True,
            "same_source_market_rows_for_candidate_and_baseline": True,
            "result_dependent_extension_allowed": False,
            "minimum_accepted_candidate_support": None,
            "support_mode": "full_window_paired_no_minimum_accepted_support",
            "service_root": (
                "examples/v8/polymarket_live_runs/"
                "challenge-model-v8-1-attempt-002"
            ),
            "strictly_later_minimum_market_start_ts_exclusive": (
                freeze_created_ts
            ),
            "maximum_attempted_market_count": 180,
            "bounded_batch_market_count": 12,
            "maximum_batch_count": 15,
            "candidate_scoring_during_raw_capture_allowed": False,
            "settlement_finalizer_enabled_during_collection": False,
            "resolution_provider_enabled_during_collection": False,
            "operator_collection_authorization_required": True,
            "operator_collection_authorization_granted": False,
            "collection_started": False,
            "collector_pid": None,
            "attempted_market_count": 0,
            "quality_valid_market_count": 0,
            "outcomes_resolution_labels_or_pnl_opened": False,
        },
        "decision_freeze": freeze
        == {
            "candidate_decisions_frozen_before_target_access": True,
            "baseline_decisions_frozen_before_target_access": True,
            "candidate_threshold_feature_controller_and_sizing_frozen": True,
            "candidate_fixed_position_size": 1.0,
            "baseline_fixed_position_size": 0.2,
            "candidate_replacement_allowed": False,
            "result_selected_rerun_allowed": False,
            "target_used_as_decision_input": False,
        },
        "settlement": settlement
        == {
            "official_read_only_resolution_only": True,
            "all_120_markets_settled_before_evaluation": True,
            "target_access_after_decision_freeze_and_market_close": True,
            "single_use_target_access_claim_required": True,
            "source_outcome_blind_rows_mutated": False,
            "costs_subtracted_exactly_once": True,
        },
        "target_mapping": mapping
        == {
            "candidate_trade_value_field": (
                "runtime_policy_after_cost_net_pnl_per_contract"
            ),
            "candidate_position_size": 1.0,
            "baseline_position_size": 0.2,
            "no_trade_after_cost_pnl": 0.0,
            "comparison_unit": "market_id",
            "all_120_markets_included": True,
        },
        "paired_gate": paired
        == {
            "scope": "all_120_markets",
            "method": "paired_market_percentile_bootstrap",
            "resample_count": 10000,
            "seed": 26212001,
            "lower_confidence_bound_quantile": 0.025,
            "candidate_minus_baseline_lcb_minimum_exclusive": 0.0,
        },
        "absolute_gate": absolute
        == {
            "scope": "all_120_markets",
            "method": "market_percentile_bootstrap",
            "resample_count": 10000,
            "seed": 26212002,
            "lower_confidence_bound_quantile": 0.025,
            "candidate_lcb_minimum_exclusive": 0.0,
        },
        "robustness": robustness.get(
            "largest_winner_removed_candidate_pnl_minimum_exclusive"
        )
        == 0.0
        and halves
        == {
            "first_half_market_count": 60,
            "second_half_market_count": 60,
            "method": "chronological_equal_halves",
            "bootstrap_method": "market_percentile_bootstrap",
            "resample_count": 10000,
            "first_half_seed": 26212003,
            "second_half_seed": 26212004,
            "upper_confidence_bound_quantile": 0.975,
            "upper_confidence_bound_minimum_inclusive": 0.0,
        },
        "support": support
        == {
            "hard_gate": False,
            "minimum_accepted_candidate_support": None,
            "all_market_rows_remain_in_paired_gate": True,
        },
        "concentration": concentration
        == {
            "hard_gate": False,
            "report_selected_action_distribution": True,
            "report_selected_side_distribution": True,
            "report_largest_absolute_single_market_pnl_share": True,
            "report_largest_winner_share_of_positive_pnl": True,
        },
        "evaluation": evaluation.get("single_use") is True
        and evaluation.get("all_hard_gates_required_for_success") is True
        and evaluation.get("protocol_isomorphic_to_historical_standard")
        is True
        and evaluation.get("manual_code_change_after_target_access_allowed")
        is False,
        "alpha": alpha
        == {
            "promotion_attempt_number": 2,
            "one_sided_alpha": 0.025,
            "confidence_level": 0.975,
            "candidate_count": 1,
            "attempt_consumed_at": "first_single_use_target_access_claim",
            "attempt_consumed": False,
        },
        "promotion": promotion
        == {
            "historical_results_eligible": False,
            "attempt_002_future_window_only": True,
            "eligible_only_if_all_hard_gates_pass": True,
            "promotion_audit_required_after_gate_pass": True,
            "automatic_promotion_allowed": False,
        },
        "safety": protocol.get("safety") == SAFE_FALSES,
    }
    _raise_failed_checks("attempt-002 preregistration", checks)


def evaluate_attempt_002_future_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate one frozen 120-market future window without mutation."""

    validate_attempt_002_preregistration(protocol)
    expected_count = int(
        protocol["future_window"]["exact_quality_valid_market_count"]
    )
    if len(rows) != expected_count:
        raise ChallengeAttempt002Error(
            f"future window must contain exactly {expected_count} rows"
        )
    normalized = [_normalize_row(row, index) for index, row in enumerate(rows)]
    market_ids = [row["market_id"] for row in normalized]
    starts = [row["market_start_ts"] for row in normalized]
    if (
        len(set(market_ids)) != expected_count
        or any(
            left >= right
            for left, right in zip(starts, starts[1:], strict=False)
        )
    ):
        raise ChallengeAttempt002Error(
            "future rows must be unique and strictly chronological"
        )

    candidate_pnl = [row["candidate_after_cost_pnl"] for row in normalized]
    baseline_pnl = [row["baseline_after_cost_pnl"] for row in normalized]
    paired_delta = [
        candidate - baseline
        for candidate, baseline in zip(
            candidate_pnl,
            baseline_pnl,
            strict=True,
        )
    ]
    evaluation = protocol["evaluation"]
    paired_spec = evaluation["full_window_paired_gate"]
    absolute_spec = evaluation["absolute_candidate_gate"]
    paired_lcb = _quantile(
        _bootstrap_sums(
            paired_delta,
            resample_count=int(paired_spec["resample_count"]),
            seed=int(paired_spec["seed"]),
        ),
        float(paired_spec["lower_confidence_bound_quantile"]),
    )
    absolute_lcb = _quantile(
        _bootstrap_sums(
            candidate_pnl,
            resample_count=int(absolute_spec["resample_count"]),
            seed=int(absolute_spec["seed"]),
        ),
        float(absolute_spec["lower_confidence_bound_quantile"]),
    )

    halves = evaluation["robustness_gates"]["chronological_halves"]
    first_count = int(halves["first_half_market_count"])
    first_pnl = candidate_pnl[:first_count]
    second_pnl = candidate_pnl[first_count:]
    first_ucb = _quantile(
        _bootstrap_sums(
            first_pnl,
            resample_count=int(halves["resample_count"]),
            seed=int(halves["first_half_seed"]),
        ),
        float(halves["upper_confidence_bound_quantile"]),
    )
    second_ucb = _quantile(
        _bootstrap_sums(
            second_pnl,
            resample_count=int(halves["resample_count"]),
            seed=int(halves["second_half_seed"]),
        ),
        float(halves["upper_confidence_bound_quantile"]),
    )

    largest_winner = max(candidate_pnl)
    largest_winner_removed = sum(candidate_pnl) - largest_winner
    side_distribution = Counter(row["candidate_side"] for row in normalized)
    action_distribution = Counter(
        row["candidate_action"] for row in normalized
    )
    absolute_pnl_sum = sum(abs(value) for value in candidate_pnl)
    positive_pnl_sum = sum(max(value, 0.0) for value in candidate_pnl)
    largest_absolute_share = (
        max(abs(value) for value in candidate_pnl) / absolute_pnl_sum
        if absolute_pnl_sum
        else 0.0
    )
    largest_winner_share = (
        largest_winner / positive_pnl_sum if positive_pnl_sum else 0.0
    )
    checks = {
        "exact_120_unique_chronological_future_markets": True,
        "all_market_rows_in_full_window_pair": True,
        "paired_bootstrap_97_5_lcb_positive": paired_lcb > 0.0,
        "candidate_absolute_bootstrap_97_5_lcb_positive": absolute_lcb
        > 0.0,
        "largest_winner_removed_candidate_pnl_positive": (
            largest_winner_removed > 0.0
        ),
        "first_half_not_significantly_negative": first_ucb >= 0.0,
        "second_half_not_significantly_negative": second_ucb >= 0.0,
        "minimum_candidate_support_not_required": True,
        "single_use_future_evidence_only": True,
        "all_safety_unlocks_remain_false": True,
    }
    all_passed = all(checks.values())
    accepted = [
        row for row in normalized if row["candidate_action"] != NO_TRADE
    ]
    return {
        "schema_version": ATTEMPT_002_RESULT_SCHEMA_VERSION,
        "attempt_id": protocol["attempt_id"],
        "candidate_id": CANDIDATE_ID,
        "baseline_id": BASELINE_ID,
        "market_count": expected_count,
        "metrics": {
            "accepted_market_count": len(accepted),
            "acceptance_rate": len(accepted) / expected_count,
            "candidate_total_after_cost_pnl": sum(candidate_pnl),
            "baseline_total_after_cost_pnl": sum(baseline_pnl),
            "candidate_minus_baseline_total_after_cost_pnl": sum(
                paired_delta
            ),
            "candidate_largest_winner_after_cost_pnl": largest_winner,
            "candidate_largest_winner_removed_after_cost_pnl": (
                largest_winner_removed
            ),
            "first_half_candidate_total_after_cost_pnl": sum(first_pnl),
            "second_half_candidate_total_after_cost_pnl": sum(second_pnl),
        },
        "bootstrap": {
            "paired_delta_97_5_lcb": paired_lcb,
            "candidate_absolute_97_5_lcb": absolute_lcb,
            "first_half_candidate_97_5_ucb": first_ucb,
            "second_half_candidate_97_5_ucb": second_ucb,
        },
        "concentration_diagnostics": {
            "hard_gate": False,
            "selected_side_distribution": dict(
                sorted(side_distribution.items())
            ),
            "selected_action_distribution": dict(
                sorted(action_distribution.items())
            ),
            "largest_absolute_single_market_pnl_share": (
                largest_absolute_share
            ),
            "largest_winner_share_of_positive_pnl": largest_winner_share,
        },
        "checks": checks,
        "all_future_success_criteria_passed": all_passed,
        "promotion_evidence_eligible": all_passed,
        "promotion_audit_required": True,
        "automatic_promotion_allowed": False,
        "historical_development_result_used_as_promotion_evidence": False,
        "safety": SAFE_FALSES,
    }


def _normalize_row(
    row: Mapping[str, Any],
    index: int,
) -> dict[str, Any]:
    market_id = str(row.get("market_id") or "")
    candidate_action = str(row.get("candidate_action") or "")
    baseline_action = str(row.get("baseline_action") or "")
    candidate_side = str(row.get("candidate_side") or "")
    baseline_side = str(row.get("baseline_side") or "")
    candidate_pnl = _finite_float(
        row.get("candidate_after_cost_pnl"),
        field=f"row {index} candidate PnL",
    )
    baseline_pnl = _finite_float(
        row.get("baseline_after_cost_pnl"),
        field=f"row {index} baseline PnL",
    )
    candidate_selected = candidate_action != NO_TRADE
    baseline_selected = baseline_action != NO_TRADE
    expected_candidate_size = 1.0 if candidate_selected else 0.0
    expected_baseline_size = 0.2 if baseline_selected else 0.0
    checks = {
        "market": bool(market_id),
        "time": _positive_integer(row.get("market_start_ts")),
        "candidate_action": _side(candidate_action) == candidate_side,
        "baseline_action": _side(baseline_action) == baseline_side,
        "candidate_no_trade": candidate_selected
        or _float_equal(candidate_pnl, 0.0),
        "baseline_no_trade": baseline_selected
        or _float_equal(baseline_pnl, 0.0),
        "candidate_size": _float_equal(
            row.get("candidate_fixed_position_size"),
            1.0,
        )
        and _float_equal(
            row.get("candidate_position_size"),
            expected_candidate_size,
        ),
        "baseline_size": _float_equal(
            row.get("baseline_fixed_position_size"),
            0.2,
        )
        and _float_equal(
            row.get("baseline_position_size"),
            expected_baseline_size,
        ),
        "freeze": row.get("candidate_decision_frozen_before_target_access")
        is True
        and row.get("baseline_decision_frozen_before_target_access") is True,
        "target": row.get("target_used_as_decision_input") is False
        and row.get("settled_after_market_close") is True
        and row.get("same_settled_market_for_candidate_and_baseline")
        is True,
        "safety": row.get("safety") == SAFE_FALSES,
    }
    _raise_failed_checks(f"future comparison row {index}", checks)
    return {
        "market_id": market_id,
        "market_start_ts": int(row["market_start_ts"]),
        "candidate_action": candidate_action,
        "candidate_side": candidate_side,
        "candidate_after_cost_pnl": candidate_pnl,
        "baseline_action": baseline_action,
        "baseline_side": baseline_side,
        "baseline_after_cost_pnl": baseline_pnl,
    }


def _bootstrap_sums(
    values: Sequence[float],
    *,
    resample_count: int,
    seed: int,
) -> list[float]:
    if not values or resample_count <= 0:
        raise ChallengeAttempt002Error("bootstrap inputs are invalid")
    rng = random.Random(seed)
    count = len(values)
    return [
        sum(values[rng.randrange(count)] for _ in range(count))
        for _ in range(resample_count)
    ]


def _quantile(values: Sequence[float], quantile: float) -> float:
    if not values or not 0.0 <= quantile <= 1.0:
        raise ChallengeAttempt002Error("quantile inputs are invalid")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _side(action: str) -> str:
    if action == NO_TRADE:
        return "NONE"
    return TRADE_ACTIONS.get(action, "")


def _finite_float(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ChallengeAttempt002Error(f"{field} is not numeric") from error
    if not math.isfinite(number):
        raise ChallengeAttempt002Error(f"{field} must be finite")
    return number


def _positive_integer(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int)
        and value > 0
    )


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


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _is_git_commit(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _raise_failed_checks(label: str, checks: Mapping[str, bool]) -> None:
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ChallengeAttempt002Error(
            f"{label} invalid: {','.join(blockers)}"
        )


__all__ = [
    "ATTEMPT_002_PREREGISTRATION_SCHEMA_VERSION",
    "ATTEMPT_002_RESULT_SCHEMA_VERSION",
    "BASELINE_ID",
    "CANDIDATE_ID",
    "ChallengeAttempt002Error",
    "evaluate_attempt_002_future_rows",
    "validate_attempt_002_preregistration",
]
