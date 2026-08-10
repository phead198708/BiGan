from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from bigan.v8.polymarket.challenge_development_lane import sha256_file
from bigan.v8.polymarket.cost_aware_residual_v4 import (
    AUTHORIZATION_INSTRUCTION,
    AUTHORIZATION_INSTRUCTION_SHA256,
    DEFAULT_AUTHORIZATION,
    DEFAULT_PROTOCOL,
    DEFAULT_REGISTRY,
    IMMUTABLE_GATE_IMPLEMENTATION,
    LINEAGE_ID,
    blend_probability_action_values,
    prequential_expert_weights,
    require_v4_candidate_implementation_binding,
    validate_residual_v4_protocol,
    validate_v4_lineage_authorization,
    verify_frozen_residual_v4_oof,
)
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.regime_adaptive_lineage import REPO_ROOT


def test_v4_exact_user_authorization_and_parent_boundary_validate() -> None:
    before = {
        "authorization": sha256_file(DEFAULT_AUTHORIZATION),
        "registry": sha256_file(DEFAULT_REGISTRY),
        "parent_terminal": sha256_file(
            REPO_ROOT
            / "examples/v8/polymarket_configs/"
            "BTC-15M-cost-aware-market-residual-v3/"
            "residual_v3_development_terminal_review.json"
        ),
        "parent_binding_audit": sha256_file(
            REPO_ROOT
            / "examples/v8/polymarket_configs/"
            "BTC-15M-cost-aware-market-residual-v3/"
            "residual_v3_frozen_artifact_binding_audit.json"
        ),
    }
    result = validate_v4_lineage_authorization()
    after = {
        "authorization": sha256_file(DEFAULT_AUTHORIZATION),
        "registry": sha256_file(DEFAULT_REGISTRY),
        "parent_terminal": sha256_file(
            REPO_ROOT
            / "examples/v8/polymarket_configs/"
            "BTC-15M-cost-aware-market-residual-v3/"
            "residual_v3_development_terminal_review.json"
        ),
        "parent_binding_audit": sha256_file(
            REPO_ROOT
            / "examples/v8/polymarket_configs/"
            "BTC-15M-cost-aware-market-residual-v3/"
            "residual_v3_frozen_artifact_binding_audit.json"
        ),
    }
    assert before == after
    assert result["authorization_valid"] is True
    assert result["lineage_id"] == LINEAGE_ID
    assert result["maximum_total_slots"] == 2
    assert result["actual_executing_module_binding_required"] is True
    assert result["safety"] == SAFETY
    assert hashlib.sha256(AUTHORIZATION_INSTRUCTION.encode("utf-8")).hexdigest() == (
        AUTHORIZATION_INSTRUCTION_SHA256
    )


def test_v4_authorization_tamper_is_fail_closed(tmp_path: Path) -> None:
    payload = json.loads(DEFAULT_AUTHORIZATION.read_text(encoding="utf-8"))
    payload["authorization_scope"]["candidate_slot_budget"][
        "maximum_total_slots"
    ] = 3
    authorization = tmp_path / "lineage_authorization.json"
    _write_frozen_json(authorization, payload)
    with pytest.raises(ValueError, match="authorization_scope"):
        validate_v4_lineage_authorization(
            authorization_path=authorization,
            registry_path=DEFAULT_REGISTRY,
        )


def test_v4_candidate_descriptor_exactly_binds_executing_module() -> None:
    module_path = (
        REPO_ROOT / "src/bigan/v8/polymarket/cost_aware_residual_v4.py"
    )
    expected = {
        "path": module_path.relative_to(REPO_ROOT).as_posix(),
        "sha256": sha256_file(module_path),
    }
    payload = {"inputs": {"candidate_implementation": expected}}
    assert require_v4_candidate_implementation_binding(payload) == expected


def test_v4_valid_but_unrelated_repository_file_swap_is_rejected() -> None:
    payload = {
        "inputs": {"candidate_implementation": dict(IMMUTABLE_GATE_IMPLEMENTATION)}
    }
    with pytest.raises(
        ValueError,
        match="does not identify the executing module",
    ):
        require_v4_candidate_implementation_binding(payload)


def test_frozen_primary_protocol_validates_and_exactly_binds_current_module() -> None:
    payload = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    validate_residual_v4_protocol(payload)
    assert sha256_file(DEFAULT_PROTOCOL) == (
        "05298c079f9e48aa412427ecc2c4cbc704885cd6d9e19ef584d6c80082362b96"
    )
    assert payload["inputs"]["candidate_implementation"] == {
        "path": "src/bigan/v8/polymarket/cost_aware_residual_v4.py",
        "sha256": "c54bffa58bda1b5a3480ba463cb06dea69b245137079b7ce728fda2bff5860da",
    }
    assert payload["candidate_budget"]["maximum_total_slots"] == 2
    assert payload["candidate_budget"]["slots_consumed_before_run"] == 0
    assert payload["state"]["training_started"] is False
    assert payload["safety"] == SAFETY


def test_frozen_primary_protocol_rejects_valid_gate_file_as_candidate() -> None:
    payload = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    payload["inputs"]["candidate_implementation"] = dict(
        IMMUTABLE_GATE_IMPLEMENTATION
    )
    with pytest.raises(ValueError, match="candidate_implementation_exact_binding"):
        validate_residual_v4_protocol(payload, verify_artifacts=False)


def test_prequential_weights_start_equal_and_use_only_supplied_history() -> None:
    empty = {
        "probability_residual": [],
        "logit_offset_binomial": [],
    }
    assert prequential_expert_weights(empty) == {
        "probability_residual": 0.5,
        "logit_offset_binomial": 0.5,
    }
    prior_only = {
        "probability_residual": [0.1, 0.2, 0.1, 0.2],
        "logit_offset_binomial": [0.4, 0.3, 0.4, 0.3],
    }
    first = prequential_expert_weights(prior_only)
    second = prequential_expert_weights(deepcopy(prior_only))
    assert first == second
    assert first["probability_residual"] > first["logit_offset_binomial"]
    assert sum(first.values()) == pytest.approx(1.0, abs=1e-15)


def test_prequential_weight_history_mismatch_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="history length mismatch"):
        prequential_expert_weights(
            {
                "probability_residual": [0.1],
                "logit_offset_binomial": [],
            }
        )


def test_convex_probability_blend_preserves_pair_coherence_and_frozen_costs() -> None:
    rows = [
        _synthetic_row("UP", entry_ask=0.55),
        _synthetic_row("DOWN", entry_ask=0.45),
    ]
    residual = [
        {"predicted_probability": 0.6},
        {"predicted_probability": 0.4},
    ]
    logit = [
        {"predicted_probability": 0.8},
        {"predicted_probability": 0.2},
    ]
    actions = blend_probability_action_values(
        rows,
        residual,
        logit,
        weights={
            "probability_residual": 0.25,
            "logit_offset_binomial": 0.75,
        },
    )
    assert [row["predicted_probability"] for row in actions] == pytest.approx(
        [0.75, 0.25]
    )
    assert sum(row["predicted_probability"] for row in actions) == pytest.approx(1.0)
    assert actions[0]["action_value"] == pytest.approx(0.19)
    assert actions[1]["action_value"] == pytest.approx(-0.21)


def test_v4_all_safety_permissions_remain_false() -> None:
    assert all(value is False for value in SAFETY.values())


def test_frozen_primary_oof_rebuilds_fail_closed_without_unlock() -> None:
    result = verify_frozen_residual_v4_oof()
    assert result["verification_passed"] is True
    assert result["all_gates_passed"] is False
    assert result["failed_gates"] == [
        "every_chronological_block_paired_delta_total_gte_zero",
        "prospective_power_required_market_count_lte_2000",
    ]
    assert result["remaining_candidate_slots"] == 1
    assert result["oof_market_count"] == 600
    assert result["manifest_sha256"] == (
        "219e87c5929760ea476f18e9b1936bec96bc9b560f73c11c65655d26da6217b8"
    )
    assert result["actual_executing_module_binding_verified"] is True
    assert result["safety"] == SAFETY


def _synthetic_row(side: str, *, entry_ask: float) -> dict:
    return {
        "market_id": "synthetic-market",
        "decision_ts": 123,
        "side": side,
        "cost_decomposition": {
            "entry_ask": entry_ask,
            "total_cost_excluding_entry_ask": 0.01,
        },
    }


def _write_frozen_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    path.with_suffix(".sha256").write_text(
        hashlib.sha256(path.read_bytes()).hexdigest() + "\n",
        encoding="utf-8",
    )
