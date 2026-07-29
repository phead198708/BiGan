from __future__ import annotations

import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.regime_adaptive_lineage import (
    FROZEN_PROTOCOL_FILES,
    LINEAGE_ID,
    SAFETY,
    validate_frozen_protocol_graph,
    verify_frozen_json,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = (
    REPO_ROOT
    / "examples"
    / "v8"
    / "polymarket_configs"
    / "BTC-15M-regime-adaptive-v1"
)


def test_frozen_regime_adaptive_protocol_graph_is_coherent() -> None:
    payloads = validate_frozen_protocol_graph(CONFIG_DIR)

    assert tuple(payloads) == FROZEN_PROTOCOL_FILES
    assert all(payload["lineage_id"] == LINEAGE_ID for payload in payloads.values())
    assert all(payload["safety"] == SAFETY for payload in payloads.values())


def test_frozen_json_sha_mismatch_fails_closed(tmp_path: Path) -> None:
    artifact = tmp_path / "protocol.json"
    artifact.write_text('{"lineage_id":"BTC-15M-regime-adaptive-v1"}\n')
    artifact.with_suffix(".sha256").write_text("0" * 64 + "\n")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_frozen_json(artifact)


def test_parent_lineage_remains_immutable_negative_evidence() -> None:
    protocol = verify_frozen_json(CONFIG_DIR / "regime_adaptive_model_protocol.json")
    lineage = verify_frozen_json(CONFIG_DIR / "lineage_manifest.json")

    assert protocol["parent_lineage"]["status"] == (
        "immutable_negative_development_evidence"
    )
    assert protocol["parent_lineage"]["development_only_forever"] is True
    assert protocol["parent_lineage"]["promotion_evidence_eligible"] is False
    assert protocol["lineage_boundary"] == {
        "existing_failed_model_lineage_may_be_modified": False,
        "existing_model_hyperparameter_tuning_may_continue": False,
        "previous_oof_result_may_be_reopened_as_validation": False,
        "previous_validation_or_oos_artifacts_allowed_as_future_validation": False,
        "parent_artifacts_allowed_for_phase_1_diagnostics_only": True,
        "new_lineage_artifacts_must_use_new_ids_and_paths": True,
        "fresh_strictly_later_evidence_required_for_any_validation_claim": True,
    }
    assert lineage["current_state"] == {
        "phase_0_complete": True,
        "phase_1_complete": False,
        "model_training_started": False,
        "fresh_collection_started": False,
        "fresh_outcomes_opened": False,
    }


def test_phase_1_diagnostic_is_descriptive_and_matches_parent_failure() -> None:
    report = verify_frozen_json(CONFIG_DIR / "temporal_drift_diagnostic_report.json")

    assert report["model_training_started"] is False
    assert report["candidate_evaluation_started"] is False
    assert report["parent_used_as_validation"] is False
    assert report["promotion_evidence_eligible"] is False
    assert report["population"]["parent_oof_market_count"] == 73
    assert report["population"]["target_or_future_label_leakage_count"] == 0
    halves = report["temporal_diagnostics"]["chronological_half"]
    assert halves["first"]["total_unit_net_pnl"] == pytest.approx(-2.42875)
    assert halves["second"]["total_unit_net_pnl"] == pytest.approx(3.63575)
    assert report["feature_coverage"]["total_recent_trade_volume"]["coverage"] == (
        pytest.approx(13 / 73)
    )
    assert report["diagnostic_conclusion"]["claim_strength"] == (
        "descriptive_hypothesis_generation_only"
    )


def test_feature_contract_is_causal_side_symmetric_and_missing_safe() -> None:
    contract = verify_frozen_json(CONFIG_DIR / "regime_feature_contract.json")

    assert contract["causality_contract"][
        "available_at_ts_must_be_lte_decision_ts"
    ]
    assert contract["causality_contract"]["max_input_ts_must_be_lte_decision_ts"]
    assert contract["causality_contract"]["market_horizon_seconds"] == 900
    assert not contract["causality_contract"][
        "missing_may_be_encoded_as_numeric_zero"
    ]
    assert contract["side_symmetry"]["shared_model_required"] is True
    assert contract["side_symmetry"]["side_identity_feature_allowed"] is False
    assert contract["side_symmetry"]["complement_proxy_allowed"] is False
    forbidden = set(contract["forbidden_features"])
    assert {"settlement_outcome", "post_decision_price", "realized_pnl", "target"} <= (
        forbidden
    )


def test_candidate_and_confirmatory_budgets_are_hard_bounded() -> None:
    family = verify_frozen_json(CONFIG_DIR / "candidate_family_protocol.json")
    budget = verify_frozen_json(CONFIG_DIR / "candidate_budget_protocol.json")

    assert len(family["candidates"]) == 5
    assert family["candidate_budget"]["maximum_candidates"] == 5
    assert family["candidate_budget"]["open_ended_search_allowed"] is False
    assert family["candidate_budget"]["threshold_grid_search_allowed"] is False
    assert budget["confirmatory_budget"]["maximum_confirmatory_rounds"] == 2
    assert budget["confirmatory_budget"]["both_rounds_required"] is True
    assert budget["confirmatory_budget"]["round_replacement_allowed"] is False
    assert budget["fresh_collection_authorization"] == {
        "status": "not_authorized",
        "authorized_attempt_cap": 0,
        "collection_started": False,
        "authorization_must_pin_candidate_and_collector_protocol": True,
        "authorization_must_precede_target_access": True,
        "extension_requires_new_pre_outcome_frozen_artifact": True,
    }


def test_collection_and_training_remain_closed() -> None:
    evaluation = verify_frozen_json(
        CONFIG_DIR / "rolling_origin_evaluation_protocol.json"
    )
    family = verify_frozen_json(CONFIG_DIR / "candidate_family_protocol.json")

    assert family["state"]["training_started"] is False
    assert family["state"]["fresh_validation_started"] is False
    assert evaluation["state"]["development_evaluation_started"] is False
    assert evaluation["state"]["fresh_confirmation_started"] is False
    assert evaluation["fresh_confirmation"]["attempt_cap"] == {
        "status": "pending_explicit_authorization",
        "authorized_attempts": 0,
        "collection_may_start": False,
        "extension_requires_new_frozen_authorization_artifact": True,
    }


def test_all_sha_sidecars_contain_plain_lowercase_digest() -> None:
    for filename in FROZEN_PROTOCOL_FILES:
        sidecar = (CONFIG_DIR / filename).with_suffix(".sha256")
        digest = sidecar.read_text(encoding="utf-8").strip()
        assert len(digest) == 64
        assert digest == digest.lower()
        assert all(character in "0123456789abcdef" for character in digest)


def test_every_frozen_json_is_canonical_json_object() -> None:
    for filename in FROZEN_PROTOCOL_FILES:
        payload = json.loads((CONFIG_DIR / filename).read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
