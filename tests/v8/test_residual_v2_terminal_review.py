from __future__ import annotations

from copy import deepcopy

import pytest

from bigan.v8.polymarket.cost_aware_residual import _load_json
from bigan.v8.polymarket.cost_aware_residual_v2 import DEFAULT_OUTPUT_DIR
from bigan.v8.polymarket.cost_aware_residual_v2_uncertainty import (
    DEFAULT_CHALLENGER_OUTPUT_DIR,
)
from bigan.v8.polymarket.residual_v2_terminal_review import (
    build_residual_v2_terminal_review,
    verify_residual_v2_terminal_review,
)


def _reports() -> tuple[dict, dict]:
    primary = _load_json(DEFAULT_OUTPUT_DIR / "residual_v2_oof_report.json")
    challenger = _load_json(
        DEFAULT_CHALLENGER_OUTPUT_DIR / "residual_v2_uncertainty_oof_report.json"
    )
    return primary, challenger


def _descriptors() -> dict[str, dict[str, str]]:
    return {
        name: {"path": f"examples/{name}.json", "sha256": "a" * 64}
        for name in (
            "primary_report",
            "primary_manifest",
            "challenger_report",
            "challenger_manifest",
            "parent_v1_terminal_review",
            "implementation",
        )
    }


def test_terminal_review_blocks_all_later_phases_without_gate_waiver() -> None:
    primary, challenger = _reports()
    review = build_residual_v2_terminal_review(
        primary=primary,
        challenger=challenger,
        source_descriptors=_descriptors(),
    )
    assert review["phase_1_terminal_failed"] is True
    assert review["candidate_budget_exhausted"] is True
    assert review["candidate_selected"] is None
    assert review["candidate_freeze_allowed"] is False
    assert review["live_shadow_start_allowed"] is False
    assert review["fresh_confirmatory_collection_authorized"] is False
    assert review["best_candidate_required_prospective_market_count"] == 2764
    assert review["best_candidate_fast_track_maximum_market_count"] == 2000
    assert all(value is False for value in review["safety"].values())


def test_terminal_review_rejects_a_waived_primary_power_gate() -> None:
    primary, challenger = _reports()
    changed = deepcopy(primary)
    changed["all_gates_passed"] = True
    changed["failed_gates"] = []
    changed["candidate_freeze_allowed"] = True
    with pytest.raises(ValueError, match="primary_result"):
        build_residual_v2_terminal_review(
            primary=changed,
            challenger=challenger,
            source_descriptors=_descriptors(),
        )


def test_terminal_review_rejects_any_third_candidate_budget() -> None:
    primary, challenger = _reports()
    changed = deepcopy(challenger)
    changed["candidate_budget_exhausted"] = False
    changed["additional_candidate_allowed"] = True
    with pytest.raises(ValueError, match="challenger_result"):
        build_residual_v2_terminal_review(
            primary=primary,
            challenger=changed,
            source_descriptors=_descriptors(),
        )


def test_frozen_terminal_review_rebuilds_and_keeps_every_permission_false() -> None:
    result = verify_residual_v2_terminal_review()
    assert result["verification_passed"] is True
    assert result["phase_1_terminal_failed"] is True
    assert result["candidate_budget_exhausted"] is True
    assert result["candidate_freeze_allowed"] is False
    assert result["live_shadow_start_allowed"] is False
    assert result["fresh_confirmatory_collection_authorized"] is False
    assert all(value is False for value in result["safety"].values())
