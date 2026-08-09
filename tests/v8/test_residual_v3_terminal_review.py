from __future__ import annotations

from copy import deepcopy

import pytest

from bigan.v8.polymarket.cost_aware_residual import _load_json
from bigan.v8.polymarket.cost_aware_residual_v3 import DEFAULT_OUTPUT_DIR
from bigan.v8.polymarket.cost_aware_residual_v3_logit import (
    DEFAULT_CHALLENGER_OUTPUT_DIR,
)
from bigan.v8.polymarket.residual_v3_terminal_review import (
    build_residual_v3_terminal_review,
)


def _reports() -> tuple[dict, dict]:
    primary = _load_json(DEFAULT_OUTPUT_DIR / "residual_v3_oof_report.json")
    challenger = _load_json(
        DEFAULT_CHALLENGER_OUTPUT_DIR / "residual_v3_logit_oof_report.json"
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
            "lineage_authorization",
            "development_data_registry",
            "parent_v2_terminal_review",
            "implementation",
        )
    }


def test_v3_terminal_review_blocks_all_unapproved_later_phases() -> None:
    primary, challenger = _reports()
    review = build_residual_v3_terminal_review(
        primary=primary,
        challenger=challenger,
        source_descriptors=_descriptors(),
    )
    assert review["phase_1_terminal_failed"] is True
    assert review["candidate_budget_exhausted"] is True
    assert review["candidate_selected"] is None
    assert review["candidate_freeze_allowed"] is False
    assert review["live_shadow_start_allowed"] is False
    assert review["fresh_collection_authorized"] is False
    assert review["best_candidate_required_prospective_market_count"] == 3043
    assert review["best_candidate_fast_track_maximum_market_count"] == 2000
    assert all(value is False for value in review["safety"].values())


def test_v3_terminal_review_rejects_gate_waiver_or_third_slot() -> None:
    primary, challenger = _reports()
    waived = deepcopy(challenger)
    waived["all_gates_passed"] = True
    waived["failed_gates"] = []
    waived["candidate_freeze_allowed"] = True
    with pytest.raises(ValueError, match="challenger_result"):
        build_residual_v3_terminal_review(
            primary=primary,
            challenger=waived,
            source_descriptors=_descriptors(),
        )
    third_slot = deepcopy(challenger)
    third_slot["candidate_budget_exhausted"] = False
    third_slot["additional_candidate_allowed"] = True
    with pytest.raises(ValueError, match="challenger_result"):
        build_residual_v3_terminal_review(
            primary=primary,
            challenger=third_slot,
            source_descriptors=_descriptors(),
        )
