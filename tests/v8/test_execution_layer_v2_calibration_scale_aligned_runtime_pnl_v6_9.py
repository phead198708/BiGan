from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.training.execution_layer_v2_calibration_scale_aligned_runtime_pnl_v6_9 import (
    apply_v6_9_score_to_runtime_pnl_mapping,
    build_v6_9_scale_contract_audit,
    build_v6_9_target_free_liveness_report,
    fit_v6_9_score_to_runtime_pnl_mapping,
    validate_calibration_scale_aligned_v6_9_profile,
)
from bigan.v8.polymarket.training.execution_layer_v2_regime_emergent_pnl_v6_8 import (
    CALIBRATION_ARTIFACT_SCHEMA_VERSION as V6_8_CALIBRATION_SCHEMA_VERSION,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = (
    PROJECT_ROOT
    / "examples/v8/polymarket_configs/"
    "execution_layer_v2_calibration_scale_aligned_runtime_pnl_v6_9_profile.json"
)


def _profile() -> dict[str, object]:
    return json.loads(PROFILE_PATH.read_text())


def _runtime_row(index: int, *, role: str) -> dict[str, object]:
    side = "UP" if index % 3 == 0 else "DOWN"
    score = 0.01 + (index % 40) * 0.005
    decision_ts = (1_000_000 if role == "development_train" else 2_000_000) + index
    return {
        "market_id": f"{role}-market-{index}",
        "role": role,
        "decision_ts": decision_ts,
        "max_input_ts": decision_ts - 1,
        "market_close_ts": decision_ts + 240,
        "side": side,
        "action": f"BUY_{side}_SELL_BEFORE_CLOSE",
        "features": {"canonical_v6_2_score": score},
        "runtime_policy_after_cost_net_pnl_per_contract": -0.04 + 1.5 * score,
        "target_available_only_post_exit_or_official_resolution": True,
        "target_used_as_decision_time_input": False,
    }


def _runtime_rows() -> list[dict[str, object]]:
    return [
        *[_runtime_row(index, role="development_train") for index in range(89)],
        *[_runtime_row(index, role="development_calibration") for index in range(45)],
    ]


def _target_free_row(index: int) -> dict[str, object]:
    side = "UP" if index % 4 == 0 else "DOWN"
    decision_ts = 3_000_000 + index
    return {
        "market_id": f"future-market-{index}",
        "decision_ts": decision_ts,
        "max_input_ts": decision_ts - 1,
        "side": side,
        "action": f"BUY_{side}_SELL_BEFORE_CLOSE",
        "v6_7_base_score": 0.04 + (index % 20) * 0.005,
        "microstructure_safety_passed": True,
        "hard_execution_safety_thresholds_unchanged": True,
        "exposure_duplicate_position_and_sizing_guards_unchanged": True,
        "source_score_mutated": False,
        "labels_outcomes_resolution_or_pnl_opened": False,
    }


def _failed_v6_8_artifact() -> dict[str, object]:
    return {
        "schema_version": V6_8_CALIBRATION_SCHEMA_VERSION,
        "pooled_residual_calibration": {"upper_confidence_bound": 0.2165},
    }


def _fit_mapping(
    *, issue229_ids: set[str] | None = None
) -> dict[str, object]:
    artifact, _, _ = fit_v6_9_score_to_runtime_pnl_mapping(
        _runtime_rows(),
        issue229_market_ids=issue229_ids or {f"future-market-{i}" for i in range(120)},
        profile=_profile(),
        runtime_target_rows_descriptor={"path": "/rows", "sha256": "a" * 64},
        runtime_target_manifest_descriptor={
            "path": "/manifest",
            "sha256": "b" * 64,
        },
    )
    return artifact


def test_profile_disables_unconditional_additive_correction_and_side_quota() -> None:
    profile = _profile()
    validate_calibration_scale_aligned_v6_9_profile(profile)

    assert profile["calibration_scale_contract"][
        "unconditional_additive_ucb_correction_allowed"
    ] is False
    assert profile["target_free_liveness"]["side_count_hard_gate_enabled"] is False
    assert profile["target_free_liveness"]["minimum_unique_market_count_per_side"] is None

    drift = copy.deepcopy(profile)
    drift["calibration_scale_contract"][
        "unconditional_additive_ucb_correction_allowed"
    ] = True
    with pytest.raises(ValueError, match="scale"):
        validate_calibration_scale_aligned_v6_9_profile(drift)


def test_scale_audit_blocks_failed_v6_8_additive_contract() -> None:
    rows = [_target_free_row(index) for index in range(120)]
    audit = build_v6_9_scale_contract_audit(
        rows,
        failed_v6_8_artifact=_failed_v6_8_artifact(),
        profile=_profile(),
    )

    assert audit["direct_additive_scale_contract_passed"] is False
    assert audit["unconditional_additive_ucb_correction_allowed"] is False
    assert audit["positive_source_score_count_before_correction"] == 120
    assert audit["positive_source_score_count_after_failed_correction"] == 0
    assert "source_score_and_runtime_target_estimand_semantics_not_proven_equivalent" in audit[
        "scale_contract_blocking_reason_codes"
    ]


def test_mapping_uses_fit_targets_and_validation_only_for_fixed_validation() -> None:
    artifact = _fit_mapping()

    assert artifact["mapping_gate_passed"] is True
    assert artifact["fit_market_count"] == 89
    assert artifact["validation_market_count"] == 45
    assert artifact["validation_labels_used_for_model_fit"] is False
    assert artifact["validation_labels_used_for_threshold_selection"] is False
    assert artifact["validation_labels_used_for_fixed_mapping_validation"] is True
    assert artifact["validation_metrics"][
        "relative_mae_improvement_over_train_mean_constant"
    ] > 0.0
    assert artifact["validation_metrics"][
        "relative_mse_improvement_over_train_mean_constant"
    ] > 0.0
    assert artifact["raw_source_score_slope"] > 0.0


def test_mapping_fails_closed_on_issue229_market_overlap() -> None:
    overlap = {"development_train-market-0"}
    artifact = _fit_mapping(issue229_ids=overlap)

    assert artifact["mapping_gate_passed"] is False
    assert artifact["frozen"] is False
    assert "issue229_market_overlap_with_mapping_lineage" in artifact[
        "mapping_gate_blocking_reason_codes"
    ]


def test_target_free_liveness_passes_without_side_quota_and_keeps_safety_blocked() -> None:
    source = [_target_free_row(index) for index in range(120)]
    artifact = _fit_mapping()
    mapped = apply_v6_9_score_to_runtime_pnl_mapping(
        source,
        mapping_artifact=artifact,
    )
    audit = build_v6_9_scale_contract_audit(
        source,
        failed_v6_8_artifact=_failed_v6_8_artifact(),
        profile=_profile(),
    )
    report = build_v6_9_target_free_liveness_report(
        source,
        mapped,
        mapping_artifact=artifact,
        scale_audit=audit,
        profile=_profile(),
        implementation_commit="a" * 40,
        candidate_freeze_created_ts=4_000_000,
    )

    assert report["target_free_liveness_gate_passed"] is True
    assert report["guard_accepted_unique_market_count"] >= 40
    assert report["minimum_per_side_support_required"] is None
    assert report["side_count_hard_gate_enabled"] is False
    assert report["current_issue229_window_eligible_for_confirmatory"] is False
    assert report["current_issue229_outcomes_opened"] is False
    for key in (
        "v8_execution_handoff_allowed",
        "source_model_candidate_eligible",
        "freeze_ready",
        "promotion_evidence_eligible",
        "#134_resume_allowed",
        "#146_start_allowed",
    ):
        assert report[key] is False


def test_target_free_mapping_rejects_outcome_or_pnl_fields() -> None:
    source = [_target_free_row(index) for index in range(120)]
    source[0]["settlement_pnl"] = 1.0

    with pytest.raises(ValueError, match="forbidden target fields"):
        apply_v6_9_score_to_runtime_pnl_mapping(
            source,
            mapping_artifact=_fit_mapping(),
        )
