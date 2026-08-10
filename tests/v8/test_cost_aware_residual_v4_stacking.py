from __future__ import annotations

import json
import math
from copy import deepcopy

import numpy as np
import pytest

from bigan.v8.polymarket.challenge_development_lane import sha256_file
from bigan.v8.polymarket.cost_aware_residual_v4 import (
    IMMUTABLE_GATE_IMPLEMENTATION,
)
from bigan.v8.polymarket.cost_aware_residual_v4_stacking import (
    DEFAULT_PROTOCOL,
    META_REGULARIZATION,
    STRUCTURAL_CHANGE,
    fit_fixed_l2_logistic_stacker,
    require_stacking_candidate_implementation_binding,
    soft_stacking_action_values,
    validate_stacking_challenger_protocol,
)
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.regime_adaptive_lineage import REPO_ROOT


def test_stacking_candidate_descriptor_exactly_binds_executing_module() -> None:
    module = (
        REPO_ROOT
        / "src/bigan/v8/polymarket/cost_aware_residual_v4_stacking.py"
    )
    expected = {
        "path": module.relative_to(REPO_ROOT).as_posix(),
        "sha256": sha256_file(module),
    }
    payload = {"inputs": {"candidate_implementation": expected}}
    assert require_stacking_candidate_implementation_binding(payload) == expected


def test_stacking_rejects_valid_but_unrelated_gate_file() -> None:
    payload = {
        "inputs": {"candidate_implementation": dict(IMMUTABLE_GATE_IMPLEMENTATION)}
    }
    with pytest.raises(
        ValueError,
        match="does not identify the executing module",
    ):
        require_stacking_candidate_implementation_binding(payload)


def test_fixed_l2_stacker_is_deterministic_and_finite() -> None:
    features = [
        [1.0, -2.0, -1.5],
        [1.0, -1.0, -0.5],
        [1.0, -0.5, -1.0],
        [1.0, 0.5, 1.0],
        [1.0, 1.0, 0.5],
        [1.0, 2.0, 1.5],
    ]
    labels = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0]
    first = fit_fixed_l2_logistic_stacker(features, labels)
    second = fit_fixed_l2_logistic_stacker(features, labels)
    assert np.array_equal(first, second)
    assert np.all(np.isfinite(first))
    assert first.shape == (3,)


def test_fixed_l2_stacker_rejects_nonbinary_labels() -> None:
    with pytest.raises(ValueError, match="labels must be binary"):
        fit_fixed_l2_logistic_stacker(
            [[1.0, 0.1, 0.2], [1.0, -0.1, -0.2]],
            [0.0, 0.5],
        )


def test_soft_stacking_preserves_pair_coherence_and_cost_ordering() -> None:
    rows = [
        _synthetic_row("UP", 0.54),
        _synthetic_row("DOWN", 0.46),
    ]
    residual = [
        {"predicted_probability": 0.65},
        {"predicted_probability": 0.35},
    ]
    logit = [
        {"predicted_probability": 0.75},
        {"predicted_probability": 0.25},
    ]
    actions = soft_stacking_action_values(
        rows,
        residual,
        logit,
        coefficients=[0.0, 0.5, 0.5],
    )
    assert math.fsum(row["predicted_probability"] for row in actions) == pytest.approx(
        1.0
    )
    assert actions[0]["action_value"] > actions[1]["action_value"]
    assert actions[0]["action_value"] == pytest.approx(
        actions[0]["predicted_probability"] - 0.54 - 0.01
    )
    assert actions[1]["action_value"] == pytest.approx(
        actions[1]["predicted_probability"] - 0.46 - 0.01
    )


def test_stacking_contract_is_structural_and_search_free() -> None:
    assert STRUCTURAL_CHANGE["changed_component"] == (
        "ensemble_combiner_and_calibration_training_design"
    )
    assert STRUCTURAL_CHANGE["parameter_weight_or_threshold_search_performed"] is False
    assert STRUCTURAL_CHANGE["route_side_missingness_or_outlier_filter_added"] is False
    assert META_REGULARIZATION == 20.0
    assert all(value is False for value in SAFETY.values())


def test_frozen_stacking_protocol_exactly_binds_final_slot() -> None:
    expected_sha256 = (
        "4ffcbca46a876ed1e3611ada214625a43f61d81deacdb5284db986933c844f24"
    )
    assert sha256_file(DEFAULT_PROTOCOL) == expected_sha256
    assert DEFAULT_PROTOCOL.with_suffix(".sha256").read_text(
        encoding="utf-8"
    ).strip() == expected_sha256
    payload = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    validate_stacking_challenger_protocol(payload)
    assert payload["inputs"]["candidate_implementation"]["sha256"] == (
        "904175ff2a2272e41a903bf53798ed72df32d3dffa076e9350d7dcfb09c9ae60"
    )
    assert payload["candidate_budget"] == {
        "maximum_total_slots": 2,
        "slot_budget_may_be_increased": False,
        "slots_consumed_before_run": 1,
        "slots_remaining_after_run": 0,
        "this_slot_ordinal": 2,
    }
    assert payload["state"] == {
        "candidate_frozen": False,
        "fresh_collection_started": False,
        "fresh_outcomes_opened": False,
        "live_shadow_started": False,
        "promotion_started": False,
        "training_started": False,
    }
    assert all(value is False for value in payload["safety"].values())


def test_frozen_stacking_protocol_rejects_candidate_module_swap() -> None:
    payload = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    swapped = deepcopy(payload)
    swapped["inputs"]["candidate_implementation"] = dict(
        IMMUTABLE_GATE_IMPLEMENTATION
    )
    with pytest.raises(ValueError, match="candidate_implementation_exact_binding"):
        validate_stacking_challenger_protocol(swapped, verify_artifacts=False)


def _synthetic_row(side: str, entry_ask: float) -> dict:
    return {
        "market_id": "synthetic-market",
        "decision_ts": 123,
        "side": side,
        "cost_decomposition": {
            "entry_ask": entry_ask,
            "total_cost_excluding_entry_ask": 0.01,
        },
    }
