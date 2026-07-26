"""Preregistered iteration-3 fixed sizing overlay for the v8.1 challenger."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from bigan.v8.polymarket.challenge_historical_development import SAFE_FALSES
from bigan.v8.polymarket.challenge_v8_1_entry_price_floor import (
    CANDIDATE_ID as ENTRY_PRICE_FLOOR_CANDIDATE_ID,
)
from bigan.v8.polymarket.challenge_v8_1_entry_price_floor import (
    ENTRY_PRICE_FLOOR,
    ChallengeEntryPriceFloorError,
    build_entry_price_floor_comparison,
    materialize_entry_price_floor_decisions,
    validate_entry_price_floor_profile,
)
from bigan.v8.polymarket.contracts import canonical_json_sha256

PROFILE_SCHEMA_VERSION = (
    "bigan-v8-challenge-v8-1-entry-price-floor-sized-profile-v1"
)
CANDIDATE_DECISION_SCHEMA_VERSION = (
    "bigan-v8-challenge-v8-1-entry-price-floor-sized-decision-v1"
)
CANDIDATE_ID = "v8_1_entry_price_floor_0_30_sized_1_0"
BASE_POSITION_SIZE = 0.2
CANDIDATE_POSITION_SIZE = 1.0

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
    "entry_price_floor_module_sha256": (
        "fad23f86b7984d2b592e72f64d3104d2e1b3af7fd75ae8d9dd66b6ef97e1a1ca"
    ),
    "entry_price_floor_profile_sha256": (
        "ea54d339c3ead15188a5fe1ede947e20e8f82cb422418f34a11277633180305e"
    ),
    "exact_195_five_action_rows_sha256": (
        "134425d9f38ffdebbf72043a8b802e95bbb87eebbe7e8d39bfa5d6f8b98828f7"
    ),
    "exact_195_market_ids_sha256": (
        "fef9eda7b8dac138b88c75f96b010bd40953795b2bcf7424debf77a004e06883"
    ),
    "iteration_001_entry_file_sha256": (
        "2bbe733ee254c66b608ff68b7cbbbc976172ebac358765c9c6ceb68924098b75"
    ),
    "iteration_001_result_sha256": (
        "80baab4220f48d2cfb98d4ccd25a9b4807462691cfc13f0a40fb96f381a90782"
    ),
    "iteration_002_entry_file_sha256": (
        "89b22557bef072be4a7df437bc4f2f126a2175a0469f6562d2652bd20ff2b253"
    ),
    "iteration_002_result_sha256": (
        "f79269fa22b6fb2b140e280416177026f2b68d37922f2096c4ce301939277619"
    ),
    "phase4_execution_policy_sha256": (
        "b4b8c86ab6ecb5c37d95dc12a212aa6e22f79f1c14fb9a787d1861033482b4b5"
    ),
    "preregistration_commit": "6dd4c485a5d58e9228e7e32b456773bf8127a1d9",
    "preregistration_sha256": (
        "5317fa03d12ce66d4c0eda8d29c18603bbcb600ffc36b80cf896b823599d4b00"
    ),
    "v7_paper_runtime_script_sha256": (
        "da922c060a4cbd47c7cdffea7e00fea5a3a0e03667e49c0b0e21e8e9acc65ac2"
    ),
}


class ChallengeEntryPriceFloorSizingError(ChallengeEntryPriceFloorError):
    """Raised when the fixed-sizing overlay cannot fail closed."""


def validate_entry_price_floor_sizing_profile(
    profile: Mapping[str, Any],
) -> None:
    """Validate the exact preregistered iteration-3 sizing policy."""

    checks = {
        "identity": profile.get("schema_version") == PROFILE_SCHEMA_VERSION
        and profile.get("candidate_id") == CANDIDATE_ID,
        "lineage": profile.get("lineage") == EXPECTED_LINEAGE,
        "policy": profile.get("policy")
        == {
            "base_candidate_id": ENTRY_PRICE_FLOOR_CANDIDATE_ID,
            "base_controller_state_transition_changed": False,
            "entry_price_floor_inclusive": ENTRY_PRICE_FLOOR,
            "fixed_candidate_position_size": CANDIDATE_POSITION_SIZE,
            "position_sizing_rule": "fixed_selected_trade_size",
            "selected_trade_set_changed": False,
            "sizing_grid_search_performed": False,
            "sizing_source": (
                "existing_v7_paper_runtime_default_max_position_size_usdc"
            ),
            "threshold_or_feature_changed": False,
        },
        "development": profile.get("development_contract")
        == {
            "baseline_position_size": BASE_POSITION_SIZE,
            "candidate_position_size": CANDIDATE_POSITION_SIZE,
            "comparison_scope": (
                "all_195_markets_in_frozen_chronological_order"
            ),
            "cost_model_changed": False,
            "historical_development_only": True,
            "no_trade_after_cost_pnl": 0.0,
            "position_lifecycle_changed": False,
            "promotion_evidence_eligible": False,
        },
        "safety": profile.get("safety") == SAFE_FALSES,
    }
    _raise_failed_checks("entry-price-floor sizing profile", checks)


def materialize_entry_price_floor_sizing_decisions(
    *,
    base_guard_rows: Sequence[Mapping[str, Any]],
    five_action_rows: Sequence[Mapping[str, Any]],
    frozen_market_ids: Sequence[str],
    profile: Mapping[str, Any],
    entry_price_floor_profile: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Freeze the same five trade decisions with size 1.0 before outcomes."""

    validate_entry_price_floor_sizing_profile(profile)
    validate_entry_price_floor_profile(entry_price_floor_profile)
    base_decisions = materialize_entry_price_floor_decisions(
        base_guard_rows=base_guard_rows,
        five_action_rows=five_action_rows,
        frozen_market_ids=frozen_market_ids,
        profile=entry_price_floor_profile,
    )
    decisions = []
    for base in base_decisions:
        if (
            base.get("candidate_id") != ENTRY_PRICE_FLOOR_CANDIDATE_ID
            or base.get("historical_development_only") is not True
            or base.get("promotion_evidence_eligible") is not False
            or base.get("safety") != SAFE_FALSES
        ):
            raise ChallengeEntryPriceFloorSizingError(
                "base entry-price-floor decision lineage is invalid"
            )
        selected = base.get("selected_action") != "NO_TRADE"
        decision = {
            key: value
            for key, value in base.items()
            if key not in {"candidate_id", "decision_id", "schema_version"}
        }
        decision.update(
            {
                "schema_version": CANDIDATE_DECISION_SCHEMA_VERSION,
                "candidate_id": CANDIDATE_ID,
                "base_candidate_id": ENTRY_PRICE_FLOOR_CANDIDATE_ID,
                "source_entry_price_floor_decision_id": base["decision_id"],
                "selected_trade_set_changed": False,
                "position_sizing_changed": True,
                "base_position_size": (
                    BASE_POSITION_SIZE if selected else 0.0
                ),
                "fixed_candidate_position_size": CANDIDATE_POSITION_SIZE,
                "candidate_position_size": (
                    CANDIDATE_POSITION_SIZE if selected else 0.0
                ),
                "position_sizing_rule": "fixed_selected_trade_size",
                "sizing_grid_search_performed": False,
                "outcome_or_pnl_field_used_at_inference": False,
                "target_used_as_decision_time_input": False,
                "historical_development_only": True,
                "promotion_evidence_eligible": False,
                "safety": SAFE_FALSES,
            }
        )
        decision["decision_id"] = canonical_json_sha256(decision)
        decisions.append(decision)
    return decisions


def build_entry_price_floor_sizing_comparison(
    *,
    candidate_decisions: Sequence[Mapping[str, Any]],
    base_comparison_rows: Sequence[Mapping[str, Any]],
    base_runtime_targets: Sequence[Mapping[str, Any]],
    frozen_market_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Apply size 1.0 to the frozen selected trades after outcomes open."""

    base_rows = build_entry_price_floor_comparison(
        candidate_decisions=candidate_decisions,
        base_comparison_rows=base_comparison_rows,
        base_runtime_targets=base_runtime_targets,
        frozen_market_ids=frozen_market_ids,
    )
    targets = {}
    for target in base_runtime_targets:
        market_id = str(target.get("market_id") or "")
        if not market_id or market_id in targets:
            raise ChallengeEntryPriceFloorSizingError(
                "runtime target identity is missing or duplicated"
            )
        per_contract = _finite_float(
            target.get("runtime_policy_after_cost_net_pnl_per_contract"),
            field="runtime target per-contract PnL",
        )
        frozen_size_pnl = _finite_float(
            target.get("runtime_policy_after_cost_net_pnl_at_frozen_size"),
            field="runtime target frozen-size PnL",
        )
        if not _float_equal(
            frozen_size_pnl,
            per_contract * BASE_POSITION_SIZE,
        ):
            raise ChallengeEntryPriceFloorSizingError(
                "runtime target is not linear at the frozen base size"
            )
        targets[market_id] = target

    rows = []
    selected_target_ids = set()
    for decision, base_row in zip(
        candidate_decisions,
        base_rows,
        strict=True,
    ):
        market_id = str(decision.get("market_id") or "")
        action = str(decision.get("selected_action") or "")
        selected = action != "NO_TRADE"
        target = targets.get(market_id)
        if selected:
            if (
                target is None
                or target.get("action") != action
                or target.get("side") != decision.get("selected_side")
            ):
                raise ChallengeEntryPriceFloorSizingError(
                    "selected trade has no matching runtime target"
                )
            candidate_pnl = _finite_float(
                target.get(
                    "runtime_policy_after_cost_net_pnl_per_contract"
                ),
                field="selected trade per-contract PnL",
            ) * CANDIDATE_POSITION_SIZE
            selected_target_ids.add(str(target.get("target_row_id") or ""))
        else:
            candidate_pnl = 0.0
        baseline_pnl = _finite_float(
            base_row.get("baseline_after_cost_pnl"),
            field="v6.7 baseline PnL",
        )
        row = dict(base_row)
        row.update(
            {
                "candidate_id": CANDIDATE_ID,
                "candidate_after_cost_pnl": candidate_pnl,
                "candidate_minus_baseline_pnl": candidate_pnl
                - baseline_pnl,
                "source_candidate_decision_id": decision["decision_id"],
                "candidate_fixed_position_size": CANDIDATE_POSITION_SIZE,
                "candidate_position_size": (
                    CANDIDATE_POSITION_SIZE if selected else 0.0
                ),
                "baseline_position_size": BASE_POSITION_SIZE,
                "source_runtime_target_id": (
                    target.get("target_row_id") if selected else None
                ),
                "position_sizing_changed": True,
                "historical_development_only": True,
                "promotion_evidence_eligible": False,
                "safety": SAFE_FALSES,
            }
        )
        rows.append(row)
    if "" in selected_target_ids or len(selected_target_ids) != sum(
        row["candidate_action"] != "NO_TRADE" for row in rows
    ):
        raise ChallengeEntryPriceFloorSizingError(
            "selected runtime target identities are invalid"
        )
    return rows


def _finite_float(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ChallengeEntryPriceFloorSizingError(
            f"{field} is not numeric"
        ) from error
    if not math.isfinite(number):
        raise ChallengeEntryPriceFloorSizingError(
            f"{field} is not finite"
        )
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
        raise ChallengeEntryPriceFloorSizingError(
            f"{label} invalid: {','.join(blockers)}"
        )


__all__ = [
    "BASE_POSITION_SIZE",
    "CANDIDATE_DECISION_SCHEMA_VERSION",
    "CANDIDATE_ID",
    "CANDIDATE_POSITION_SIZE",
    "ChallengeEntryPriceFloorSizingError",
    "PROFILE_SCHEMA_VERSION",
    "build_entry_price_floor_sizing_comparison",
    "materialize_entry_price_floor_sizing_decisions",
    "validate_entry_price_floor_sizing_profile",
]
