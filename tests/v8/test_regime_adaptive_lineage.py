from __future__ import annotations

import hashlib
import json
import math
import shutil
import statistics
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path
from statistics import NormalDist

import pytest

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.moe_collection_boundary_r2 import (
    ATTEMPT_CAP as MOE_V2_R2_ATTEMPT_CAP,
)
from bigan.v8.polymarket.moe_collection_boundary_r2 import (
    CANDIDATE_BUNDLE_HASH as MOE_V2_R2_BUNDLE_HASH,
)
from bigan.v8.polymarket.moe_collection_boundary_r2 import (
    SAFETY as MOE_V2_R2_SAFETY,
)
from bigan.v8.polymarket.moe_collection_boundary_r2 import (
    STATE_BLOCKED as MOE_V2_R2_STATE_BLOCKED,
)
from bigan.v8.polymarket.moe_collection_boundary_r2 import (
    TARGET_MARKET_COUNT as MOE_V2_R2_TARGET,
)
from bigan.v8.polymarket.moe_collection_boundary_r2 import (
    AuthorizationExpectation,
    build_manual_authorization_payload,
    frozen_authorization_artifacts,
    select_exact_authorized_window,
    validate_and_authorize_exact_outcome_access,
    validate_authorized_attempt_ledger,
    validate_capture_manifest_structure,
    validate_exact_window_selection,
    validate_manual_collection_authorization,
)
from bigan.v8.polymarket.moe_confirmatory_lineage import (
    assert_metric_payload_matches,
    deterministic_moe_route,
    frozen_expert_or_fallback,
)
from bigan.v8.polymarket.moe_confirmatory_v2 import (
    BASE_COMMIT as MOE_V2_BASE_COMMIT,
)
from bigan.v8.polymarket.moe_confirmatory_v2 import (
    CANDIDATE_SAMPLE_SIZES as MOE_V2_SAMPLE_SIZES,
)
from bigan.v8.polymarket.moe_confirmatory_v2 import (
    SAFETY as MOE_V2_SAFETY,
)
from bigan.v8.polymarket.moe_confirmatory_v2 import (
    V1_BUNDLE_HASH,
    load_and_verify_v2_artifact,
    validate_v2_artifact_in_fresh_environment,
)
from bigan.v8.polymarket.moe_precollection_hardening_r1 import (
    BASE_COMMIT as MOE_V2_R1_BASE_COMMIT,
)
from bigan.v8.polymarket.moe_precollection_hardening_r1 import (
    CANDIDATE_BUNDLE_HASH as MOE_V2_R1_BUNDLE_HASH,
)
from bigan.v8.polymarket.moe_precollection_hardening_r1 import (
    FORBIDDEN_HEALTH_SNAPSHOT_FIELDS,
    assert_outcome_access_allowed,
    binomial_tail_probability,
    deterministic_exact_window,
    empirical_bootstrap_lcb_crossing_power,
    minimum_attempt_cap,
    validate_attempt_hash_chain,
    validate_exact_window,
    validate_health_snapshot,
    validate_population_reconciliation,
    verify_raw_evidence_manifest_hash,
    wilson_lower_bound,
)
from bigan.v8.polymarket.moe_static_artifact import (
    load_and_verify_static_moe_artifact,
    validate_static_moe_artifact_in_fresh_environment,
)
from bigan.v8.polymarket.regime_adaptive_candidate_evaluation import (
    _annotate_regime,
    _candidate_metrics,
    _select_candidate,
    _selected_rows,
)
from bigan.v8.polymarket.regime_adaptive_lineage import (
    FROZEN_PROTOCOL_FILES,
    LINEAGE_ID,
    SAFETY,
    validate_frozen_protocol_graph,
    verify_frozen_json,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "examples" / "v8" / "polymarket_configs" / "BTC-15M-regime-adaptive-v1"
EVALUATION_DIR = (
    REPO_ROOT
    / "examples"
    / "v8"
    / "polymarket_training_artifacts"
    / "BTC-15M-regime-adaptive-v1-development-evaluation"
)
MOE_CONFIG_DIR = (
    REPO_ROOT / "examples" / "v8" / "polymarket_configs" / "BTC-15M-MoE-confirmatory-v1"
)
MOE_V2_CONFIG_DIR = (
    REPO_ROOT / "examples" / "v8" / "polymarket_configs" / "BTC-15M-MoE-confirmatory-v2"
)
MOE_V2_R2_FIXTURE_DIR = MOE_V2_CONFIG_DIR / "collection_boundary_validation_fixture"


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

    assert protocol["parent_lineage"]["status"] == ("immutable_negative_development_evidence")
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

    assert contract["causality_contract"]["available_at_ts_must_be_lte_decision_ts"]
    assert contract["causality_contract"]["max_input_ts_must_be_lte_decision_ts"]
    assert contract["causality_contract"]["market_horizon_seconds"] == 900
    assert not contract["causality_contract"]["missing_may_be_encoded_as_numeric_zero"]
    assert contract["side_symmetry"]["shared_model_required"] is True
    assert contract["side_symmetry"]["side_identity_feature_allowed"] is False
    assert contract["side_symmetry"]["complement_proxy_allowed"] is False
    forbidden = set(contract["forbidden_features"])
    assert {"settlement_outcome", "post_decision_price", "realized_pnl", "target"} <= (forbidden)


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
    evaluation = verify_frozen_json(CONFIG_DIR / "rolling_origin_evaluation_protocol.json")
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


def test_regime_annotation_uses_frozen_decision_time_cutoffs() -> None:
    contract = verify_frozen_json(CONFIG_DIR / "regime_feature_contract.json")
    row = {
        "side": "DOWN",
        "features": {
            "signed_btc_return_15m": 0.0003,
            "btc_volatility_15m": 0.0005,
            "combined_spread_bps": 500.0,
            "selected_recent_trade_volume": float("nan"),
            "opposite_recent_trade_volume": float("nan"),
            "selected_liquidity_depth": 30_000.0,
            "opposite_liquidity_depth": 20_000.0,
        },
    }

    _annotate_regime(row, contract)

    assert row["btc_return_regime"] == "bearish"
    assert row["volatility_bucket"] == "high"
    assert row["spread_bucket"] == "medium"
    assert row["volume_bucket"] == "missing"
    assert row["depth_bucket"] == "medium"
    assert row["expert_route"] == "high_vol"


def test_action_policy_selects_earliest_positive_decision_and_allows_no_trade() -> None:
    rows = [
        {
            "market_id": "trade",
            "decision_ts": 1,
            "side": "UP",
            "selection_score": -0.01,
        },
        {
            "market_id": "trade",
            "decision_ts": 1,
            "side": "DOWN",
            "selection_score": -0.02,
        },
        {
            "market_id": "trade",
            "decision_ts": 2,
            "side": "UP",
            "selection_score": 0.02,
        },
        {
            "market_id": "trade",
            "decision_ts": 2,
            "side": "DOWN",
            "selection_score": 0.01,
        },
        {
            "market_id": "no-trade",
            "decision_ts": 1,
            "side": "UP",
            "selection_score": 0.0,
        },
        {
            "market_id": "no-trade",
            "decision_ts": 1,
            "side": "DOWN",
            "selection_score": -0.01,
        },
    ]

    selected = _selected_rows(rows)

    assert [(row["market_id"], row["decision_ts"], row["side"]) for row in selected] == [
        ("trade", 2, "UP")
    ]


def test_synthetic_40_market_evaluation_and_selection() -> None:
    evaluation = verify_frozen_json(CONFIG_DIR / "rolling_origin_evaluation_protocol.json")
    ordered_markets = [(index, f"market-{index:02d}") for index in range(40)]
    rows: list[dict[str, object]] = []
    for index, (_, market_id) in enumerate(ordered_markets):
        winning_side = "UP" if index % 2 == 0 else "DOWN"
        for side in ("UP", "DOWN"):
            wins = side == winning_side
            rows.append(
                {
                    "market_id": market_id,
                    "decision_ts": index,
                    "side": side,
                    "selection_score": 0.2 if wins else -0.2,
                    "win_probability": 0.8 if wins else 0.2,
                    "settlement_payout": 1.0 if wins else 0.0,
                    "target": 0.1 if wins else -0.9,
                    "gross_price_edge": 0.12 if wins else -0.88,
                    "entry_spread_cost": 0.01,
                    "fees": 0.002,
                    "slippage": 0.005,
                    "liquidity_impact": 0.003,
                    "btc_return_regime": "bullish" if wins else "bearish",
                    "volatility_bucket": "medium",
                    "spread_bucket": "low",
                    "volume_bucket": "missing",
                    "depth_bucket": "high",
                }
            )

    metrics = _candidate_metrics(
        candidate_id="synthetic",
        candidate_ordinal=1,
        oof_rows=rows,
        ordered_oof_markets=ordered_markets,
        evaluation=evaluation,
    )
    selection = _select_candidate([metrics], evaluation)

    assert metrics["trading_metrics"]["market_count"] == 40
    assert metrics["trading_metrics"]["accepted_market_count"] == 40
    assert metrics["trading_metrics"]["total_unit_net_pnl"] == pytest.approx(4.0)
    assert metrics["development_selection_eligible"] is True
    assert selection["selected_candidate_id"] == "synthetic"
    assert selection["fresh_collection_allowed"] is False


def test_frozen_development_result_stops_before_fresh_collection() -> None:
    result = verify_frozen_json(CONFIG_DIR / "development_evaluation_result.json")

    assert result["population"] == {
        "development_market_count": 113,
        "initial_strictly_prior_training_market_count": 40,
        "rolling_origin_evaluation_market_count": 73,
        "candidate_count": 5,
        "candidate_budget_consumed": 5,
        "candidate_budget_maximum": 5,
        "target_or_future_label_leakage_count": 0,
    }
    assert all(
        candidate["development_selection_eligible"] is False
        for candidate in result["candidate_results"]
    )
    mixture = next(
        candidate
        for candidate in result["candidate_results"]
        if candidate["candidate_id"] == "mixture_of_experts"
    )
    assert mixture["total_unit_net_pnl"] == pytest.approx(4.372)
    assert mixture["mean_unit_net_pnl_bootstrap_95pct_lower"] == pytest.approx(-0.03593672945205478)
    assert mixture["first_chronological_half_total_unit_net_pnl"] > 0.0
    assert mixture["second_chronological_half_total_unit_net_pnl"] > 0.0
    assert result["selection"]["selected_candidate_id"] is None
    assert result["selection"]["fresh_collection_allowed"] is False
    assert result["stopping_rule"]["phase_6_allowed"] is False
    assert result["stopping_rule"]["fresh_collection_started"] is False
    assert result["stopping_rule"]["fresh_outcomes_opened"] is False


def test_all_development_folds_are_strictly_prior_and_leakage_free() -> None:
    folds = [
        json.loads(line)
        for line in (EVALUATION_DIR / "development_fold_audits.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert len(folds) == 5 * 73
    assert {fold["candidate_id"] for fold in folds} == {
        "global_baseline",
        "regime_conditioned_calibration",
        "mixture_of_experts",
        "drift_aware_rolling_calibration",
        "uncertainty_aware_abstention",
    }
    assert min(fold["strictly_prior_market_count"] for fold in folds) == 40
    assert max(fold["strictly_prior_market_count"] for fold in folds) == 112
    assert all(fold["target_market_used_for_fit"] is False for fold in folds)
    assert all(fold["future_market_used_for_fit"] is False for fold in folds)
    assert all(fold["target_or_future_label_leakage_count"] == 0 for fold in folds)
    assert all(fold["promotion_evidence_eligible"] is False for fold in folds)


def test_development_predictions_are_bounded_and_never_promotion_evidence() -> None:
    predictions = [
        json.loads(line)
        for line in (EVALUATION_DIR / "development_oof_predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert len(predictions) == 5 * 73 * 4
    assert all(0.0 <= row["win_probability"] <= 1.0 for row in predictions)
    assert all(row["development_only_forever"] is True for row in predictions)
    assert all(row["promotion_evidence_eligible"] is False for row in predictions)
    assert all(row["safety"] == SAFETY for row in predictions)


def test_moe_parent_lineage_is_unchanged_and_budget_remains_consumed() -> None:
    result_path = CONFIG_DIR / "development_evaluation_result.json"
    result = verify_frozen_json(result_path)

    assert _sha256(result_path) == (
        "e2bce83faa98319c69ddd7560f8cdd6cb0a6aa983d728710c3c0ab22f7e5b57e"
    )
    assert result["population"]["candidate_budget_consumed"] == 5
    assert result["population"]["candidate_budget_maximum"] == 5
    assert result["selection"]["selected_candidate_id"] is None
    assert result["selection"]["fresh_collection_allowed"] is False
    assert result["stopping_rule"]["model_freeze_created"] is False


def test_moe_provenance_fails_closed_for_unresolvable_source_commit() -> None:
    attestation = verify_frozen_json(
        MOE_CONFIG_DIR / "development_evaluation_provenance_attestation.json"
    )

    assert attestation["original_source_commit"]["recorded_value"] == (
        "364fd65b08849cb36227a3c4bb1b55a62cc68825"
    )
    assert attestation["original_source_commit"]["exact_commit_reachable"] is False
    assert attestation["reachable_evaluator"]["commit"] == (
        "364fd65afa4908170dbc2ae5ff4f71c8a2475573"
    )
    assert attestation["timestamp_reconciliation"][
        "protocol_definitions_existed_before_target_access"
    ]
    assert attestation["attestation_status"]["passed"] is False
    assert attestation["attestation_status"]["new_candidate_freeze_allowed"] is False


def test_moe_metric_reconciliation_passes_for_all_five_candidates() -> None:
    report = verify_frozen_json(MOE_CONFIG_DIR / "development_metric_reconciliation_report.json")

    assert report["reconciliation_passed"] is True
    assert report["population"] == {
        "candidate_count": 5,
        "prediction_row_count": 1460,
        "fold_audit_count": 365,
        "oof_market_count": 73,
        "target_or_future_label_leakage_count": 0,
    }
    assert report["cost_reconciliation"]["row_decomposition_mismatch_count"] == 0
    assert all(item["comparison"]["passed"] for item in report["candidate_reconciliation"])
    assert {item["candidate_id"] for item in report["candidate_reconciliation"]} == {
        "global_baseline",
        "regime_conditioned_calibration",
        "mixture_of_experts",
        "drift_aware_rolling_calibration",
        "uncertainty_aware_abstention",
    }
    assert all(
        item["comparison"]["metric_mismatches"] == []
        and item["comparison"]["gate_mismatches"] == []
        and item["comparison"]["result_summary_mismatches"] == []
        for item in report["candidate_reconciliation"]
    )
    assert report["parent_selection_reconciliation"]["selected_candidate_id"] is None
    assert report["parent_selection_reconciliation"]["fresh_collection_allowed"] is False


def test_moe_metric_tampering_fails_closed() -> None:
    with pytest.raises(ValueError, match="metric reconciliation mismatch"):
        assert_metric_payload_matches(
            {"accepted_market_count": 72, "total_unit_net_pnl": 4.372},
            {"accepted_market_count": 72, "total_unit_net_pnl": 4.373},
        )


@pytest.mark.parametrize(
    ("volatility_bucket", "btc_return_regime", "expected"),
    [
        ("high", "bearish", "high_vol"),
        ("medium", "bullish", "bullish"),
        ("low", "bearish", "bearish"),
        ("medium", "sideways", "low_vol"),
    ],
)
def test_moe_router_is_deterministic_and_uses_frozen_precedence(
    volatility_bucket: str,
    btc_return_regime: str,
    expected: str,
) -> None:
    inputs = {
        "decision_ts": 100,
        "available_at_ts": 100,
        "max_input_ts": 99,
        "volatility_bucket": volatility_bucket,
        "btc_return_regime": btc_return_regime,
    }

    assert deterministic_moe_route(inputs) == expected
    assert deterministic_moe_route(dict(reversed(list(inputs.items())))) == expected


def test_moe_router_rejects_future_and_outcome_fields() -> None:
    causal = {
        "decision_ts": 100,
        "available_at_ts": 100,
        "max_input_ts": 99,
        "volatility_bucket": "high",
        "btc_return_regime": "bullish",
    }
    with pytest.raises(ValueError, match="causality violation"):
        deterministic_moe_route({**causal, "max_input_ts": 101})
    with pytest.raises(ValueError, match="outcome or future fields"):
        deterministic_moe_route({**causal, "resolved_outcome": "UP"})
    with pytest.raises(ValueError, match="outcome or future fields"):
        deterministic_moe_route({**causal, "target": 0.5})


def test_moe_frozen_support_boundary_selects_expert_or_fallback() -> None:
    assert (
        frozen_expert_or_fallback(
            route="high_vol",
            expert_training_market_count=19,
        )
        == "global_baseline_fallback"
    )
    assert (
        frozen_expert_or_fallback(
            route="high_vol",
            expert_training_market_count=20,
        )
        == "moe_expert_high_vol"
    )


def test_moe_attribution_totals_reconcile() -> None:
    report = verify_frozen_json(MOE_CONFIG_DIR / "moe_route_attribution_report.json")
    rows = [
        json.loads(line)
        for line in (MOE_CONFIG_DIR / "moe_route_attribution.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    accepted = [row for row in rows if row["accepted"]]

    assert len(rows) == report["market_count"] == 73
    assert len(accepted) == report["accepted_market_count"] == 72
    assert report["architecture_type"] == (
        "deterministic_regime_router_with_conditional_experts_and_global_fallback"
    )
    required_fields = {
        "market_id",
        "decision_ts",
        "router_inputs",
        "requested_route",
        "expert_id",
        "expert_training_market_count",
        "expert_available",
        "fallback_used",
        "actual_model_used",
        "prediction",
        "selected_side",
        "accepted",
        "unit_net_pnl",
        "cost_decomposition",
        "chronological_half",
        "regime_bucket",
        "provider_and_missingness",
    }
    assert all(required_fields <= row.keys() for row in rows)
    assert sum(row["unit_net_pnl"] for row in rows) == pytest.approx(report["total_unit_net_pnl"])
    assert report["pnl_attribution"]["native_expert_pnl"] + report["pnl_attribution"][
        "global_fallback_pnl"
    ] == pytest.approx(report["total_unit_net_pnl"])
    assert sum(row["selected_side"] == "UP" for row in accepted) + sum(
        row["selected_side"] == "DOWN" for row in accepted
    ) == len(accepted)
    assert report["pnl_attribution"]["chronological_half_pnl"]["first"] + report["pnl_attribution"][
        "chronological_half_pnl"
    ]["second"] == pytest.approx(report["total_unit_net_pnl"])
    provider = report["provider_and_missingness"]
    assert provider["accepted_complete_feature_market_count"] + provider[
        "accepted_missing_feature_market_count"
    ] == len(accepted)
    assert report["attribution_reconciliation_passed"] is True
    assert set(report["fallback"]["share_by_chronological_quartile"]) == {
        "q1",
        "q2",
        "q3",
        "q4",
    }
    assert set(report["support_evolution_by_route"]) == {
        "high_vol",
        "bullish",
        "bearish",
        "low_vol",
    }
    assert set(report["provider_and_missingness"]["by_route"]) == {
        "high_vol",
        "bullish",
        "bearish",
        "low_vol",
    }


def test_moe_complement_proxy_remains_forbidden() -> None:
    family = verify_frozen_json(CONFIG_DIR / "candidate_family_protocol.json")

    assert family["shared_action_policy"]["true_paired_executable_ask_required"] is True
    assert family["shared_action_policy"]["complement_proxy_allowed"] is False


def test_moe_candidate_contract_freezes_architecture_without_collection() -> None:
    candidate = verify_frozen_json(MOE_CONFIG_DIR / "moe_candidate_contract.json")

    assert candidate["candidate_id"] == "mixture_of_experts"
    assert candidate["architecture_type"] == (
        "deterministic_regime_router_with_conditional_experts_and_global_fallback"
    )
    assert candidate["hardening_inputs"]["metric_reconciliation"]["reconciliation_passed"] is True
    assert candidate["frozen_behavior"]["complement_proxy_allowed"] is False
    assert candidate["state"]["fresh_collection_authorized"] is False
    assert candidate["state"]["fresh_collection_started"] is False
    assert candidate["state"]["fresh_outcomes_opened"] is False
    assert candidate["safety"] == SAFETY


def test_moe_repository_artifact_graph_loads_in_fresh_clone(
    tmp_path: Path,
) -> None:
    graph = verify_frozen_json(MOE_CONFIG_DIR / "moe_artifact_graph.json")
    graph_sha = _sha256(MOE_CONFIG_DIR / "moe_artifact_graph.json")
    fresh_root = tmp_path / "fresh-clone"
    fresh_graph = fresh_root / (
        "examples/v8/polymarket_configs/BTC-15M-MoE-confirmatory-v1/moe_artifact_graph.json"
    )
    fresh_graph.parent.mkdir(parents=True)
    shutil.copyfile(MOE_CONFIG_DIR / "moe_artifact_graph.json", fresh_graph)
    for descriptor in graph["artifacts"].values():
        source = REPO_ROOT / descriptor["path"]
        destination = fresh_root / descriptor["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    loaded = load_and_verify_static_moe_artifact(
        graph_path=fresh_graph,
        expected_graph_sha256=graph_sha,
        repository_root=fresh_root,
    )

    assert graph["all_paths_repository_relative"] is True
    assert graph["machine_local_absolute_paths_allowed"] is False
    assert not any(
        Path(descriptor["path"]).is_absolute() for descriptor in graph["artifacts"].values()
    )
    assert loaded["all_fixtures_reproduced"] is True
    assert loaded["fallback_loaded"] is True
    assert loaded["loaded_experts"] == {
        "high_vol": "xgboost_json",
        "bullish": "xgboost_json",
        "bearish": "xgboost_json",
        "low_vol": "support_below_minimum_stub_json",
    }
    assert loaded["fallback_rounds"] == 104
    assert loaded["expert_rounds"] == {
        "high_vol": 104,
        "bullish": 104,
        "bearish": 104,
        "low_vol": 0,
    }
    assert loaded["fresh_collection_authorized"] is False
    assert loaded["safety"] == SAFETY


def test_moe_repository_artifact_runtime_is_deterministic() -> None:
    graph_path = MOE_CONFIG_DIR / "moe_artifact_graph.json"
    graph_sha = _sha256(graph_path)

    first = load_and_verify_static_moe_artifact(
        graph_path=graph_path,
        expected_graph_sha256=graph_sha,
        repository_root=REPO_ROOT,
    )
    second = load_and_verify_static_moe_artifact(
        graph_path=graph_path,
        expected_graph_sha256=graph_sha,
        repository_root=REPO_ROOT,
    )

    assert first == second
    assert first["bundle_hash"] == (
        "30d180b028c83146fafd81c8b81269f51fa567b30bc5ab4d3577dd99c256dcf8"
    )
    assert first["artifact_count"] == first["verified_child_sha256_count"] == 13
    assert first["router_contract_loaded"] is True
    assert first["feature_contract_loaded"] is True
    assert first["cost_contract_loaded"] is True
    assert first["all_fixtures_reproduced"] is True
    assert len(first["fixture_results"]) == 2
    assert all(
        len(result["raw_probabilities"]) == 2
        and len(result["normalized_probabilities"]) == 2
        and len(result["scores"]) == 2
        and result["reproduced"] is True
        for result in first["fixture_results"]
    )


def test_moe_mandatory_runtime_gate_passes_in_fresh_environment() -> None:
    graph_path = MOE_CONFIG_DIR / "moe_artifact_graph.json"
    result = validate_static_moe_artifact_in_fresh_environment(
        graph_path=graph_path,
        expected_graph_sha256=_sha256(graph_path),
        repository_root=REPO_ROOT,
    )

    assert result["runtime_validation_passed"] is True
    assert result["fresh_environment_resolution"] is True
    assert result["bundle_hash_verified"] is True
    assert result["verified_child_sha256_count"] == 13
    assert result["router_contract_loaded"] is True
    assert result["feature_contract_loaded"] is True
    assert result["all_expert_artifacts_loaded"] is True
    assert result["available_expert_models_loaded"] == [
        "bearish",
        "bullish",
        "high_vol",
    ]
    assert result["unavailable_expert_stubs_verified"] == ["low_vol"]
    assert result["fallback_loaded"] is True
    assert result["synthetic_fixture_count"] == 2
    assert result["synthetic_deterministic_prediction_equality"] is True
    assert result["fresh_collection_authorized"] is False
    assert result["safety"] == SAFETY


def test_moe_artifact_graph_sha_mismatch_fails_closed() -> None:
    with pytest.raises(ValueError, match="graph SHA-256 mismatch"):
        load_and_verify_static_moe_artifact(
            graph_path=MOE_CONFIG_DIR / "moe_artifact_graph.json",
            expected_graph_sha256="0" * 64,
            repository_root=REPO_ROOT,
        )


def test_moe_artifact_component_sha_mismatch_fails_closed(
    tmp_path: Path,
) -> None:
    graph = json.loads((MOE_CONFIG_DIR / "moe_artifact_graph.json").read_text(encoding="utf-8"))
    graph["artifacts"]["moe_global_fallback.json"]["sha256"] = "0" * 64
    graph_path = tmp_path / "tampered-graph.json"
    graph_path.write_text(
        json.dumps(graph, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="artifact SHA-256 mismatch"):
        load_and_verify_static_moe_artifact(
            graph_path=graph_path,
            expected_graph_sha256=_sha256(graph_path),
            repository_root=REPO_ROOT,
        )


def test_moe_artifact_bundle_hash_mismatch_fails_closed(
    tmp_path: Path,
) -> None:
    graph = json.loads((MOE_CONFIG_DIR / "moe_artifact_graph.json").read_text(encoding="utf-8"))
    graph["bundle_hash"] = "0" * 64
    graph_path = tmp_path / "wrong-bundle-hash-graph.json"
    graph_path.write_text(
        json.dumps(graph, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="bundle path is not content addressed"):
        load_and_verify_static_moe_artifact(
            graph_path=graph_path,
            expected_graph_sha256=_sha256(graph_path),
            repository_root=REPO_ROOT,
        )


def test_moe_collection_authorization_is_an_inactive_template() -> None:
    template = verify_frozen_json(
        MOE_CONFIG_DIR / "moe_fresh_collection_authorization_template.json"
    )
    manifest = verify_frozen_json(MOE_CONFIG_DIR / "moe_model_manifest.json")
    graph = verify_frozen_json(MOE_CONFIG_DIR / "moe_artifact_graph.json")

    assert template["artifact_role"] == ("inactive_template_only_not_collection_authority")
    assert template["template_usable_as_collection_authorization"] is False
    assert template["state"] == {
        "fresh_collection_authorized": False,
        "fresh_collection_started": False,
        "fresh_outcomes_opened": False,
    }
    assert manifest["fresh_collection_authorized"] is False
    assert manifest["fresh_collection_started"] is False
    assert manifest["fresh_outcomes_opened"] is False
    assert graph["fresh_collection_authorized"] is False
    assert graph["fresh_collection_started"] is False
    assert graph["fresh_outcomes_opened"] is False
    assert template["safety"] == manifest["safety"] == graph["safety"] == SAFETY


def test_moe_confirmatory_protocol_is_frozen_and_internally_consistent() -> None:
    protocol = verify_frozen_json(MOE_CONFIG_DIR / "moe_confirmatory_protocol.json")
    graph = verify_frozen_json(MOE_CONFIG_DIR / "moe_artifact_graph.json")
    reporting = verify_frozen_json(MOE_CONFIG_DIR / "moe_future_evaluation_reporting_contract.json")
    collector = verify_frozen_json(MOE_CONFIG_DIR / "moe_confirmatory_collector_protocol.json")

    for descriptor in (
        protocol["frozen_inputs"]["router_contract"],
        protocol["frozen_inputs"]["feature_contract"],
        protocol["frozen_inputs"]["cost_contract"],
        protocol["frozen_inputs"]["collector_protocol"],
        protocol["frozen_inputs"]["future_evaluation_reporting_contract"],
        protocol["frozen_inputs"]["baseline_behavior_contract"],
        protocol["frozen_inputs"]["artifact_runtime_validation_report"],
        protocol["frozen_inputs"]["confirmatory_power_analysis"],
    ):
        path = REPO_ROOT / descriptor["path"]
        assert path.is_file()
        assert _sha256(path) == descriptor["sha256"]
    candidate = protocol["frozen_inputs"]["candidate_artifact"]
    baseline = protocol["frozen_inputs"]["baseline_artifact"]
    assert candidate["bundle_hash"] == graph["bundle_hash"]
    assert candidate["artifact_graph_sha256"] == _sha256(MOE_CONFIG_DIR / "moe_artifact_graph.json")
    assert baseline["sha256"] == graph["artifacts"]["moe_global_fallback.json"]["sha256"]
    assert protocol["bootstrap"] == {
        "method": ("market_level_paired_percentile_bootstrap_with_shared_resample_indices"),
        "unit": "unique_market",
        "seed": 26015,
        "resamples": 10000,
        "confidence": 0.975,
        "sidedness": "one_sided_lower_confidence_bound",
        "quantile": 0.025,
        "no_trade_value": 0.0,
        "candidate_and_baseline_use_same_market_draws": True,
        "route_or_trade_level_resampling_allowed": False,
    }
    rounds = protocol["confirmatory_rounds"]
    assert rounds["number_of_confirmatory_rounds"] == 2
    assert rounds["minimum_quality_valid_outcome_finalized_markets_per_round"] == 40
    assert rounds["paired_executable_ask_coverage_minimum"] == 0.95
    stopping = protocol["multiplicity_and_stopping"]
    assert stopping["both_rounds_must_independently_pass"] is True
    assert stopping["optional_stopping_allowed"] is False
    assert stopping["failed_round_rerun_allowed"] is False
    assert stopping["route_filtering_allowed"] is False
    assert stopping["post_hoc_exclusions_allowed"] is False
    assert protocol["state"]["fresh_collection_authorized"] is False
    assert protocol["state"]["fresh_collection_started"] is False
    assert protocol["state"]["fresh_outcomes_opened"] is False
    assert protocol["pre_collection_mandatory_gates"] == {
        "artifact_runtime_validation_must_pass": True,
        "runtime_validation_report_hash_must_match": True,
        "baseline_behavior_contract_hash_must_match": True,
        "reporting_contract_hash_must_match": True,
        "manual_collection_authorization_required": True,
        "current_manual_collection_authorization_present": False,
        "collection_may_start": False,
    }
    assert (
        collector["quality_validity"][
            "paired_up_down_executable_ask_availability_is_not_a_selection_filter"
        ]
        is True
    )
    assert protocol["safety"] == reporting["safety"] == collector["safety"] == SAFETY


def test_moe_future_reporting_contract_requires_all_unfiltered_panels() -> None:
    contract = verify_frozen_json(MOE_CONFIG_DIR / "moe_future_evaluation_reporting_contract.json")
    panels = contract["required_panels"]

    assert set(panels) == {
        "overall",
        "route",
        "actual_model",
        "provider",
        "regime",
    }
    assert set(panels["overall"]["metrics"]) >= {
        "moe_total_unit_net_pnl",
        "baseline_total_unit_net_pnl",
        "delta_total_unit_net_pnl",
        "delta_market_bootstrap_lcb",
    }
    assert panels["route"]["dimensions"] == [
        "high_vol_expert",
        "bullish_expert",
        "bearish_expert",
        "low_vol_fallback",
    ]
    assert panels["actual_model"]["dimensions"] == [
        "expert_prediction",
        "fallback_prediction",
    ]
    assert panels["provider"]["dimensions"] == [
        "complete_feature_markets",
        "missingness_markets",
    ]
    assert panels["regime"]["dimensions"] == [
        "bullish",
        "bearish",
        "sideways",
    ]
    assert contract["forbidden_reporting_behavior"]["route_filtering"] is True
    assert contract["forbidden_reporting_behavior"]["post_hoc_exclusions"] is True
    assert contract["population"]["all_frozen_round_markets_required"] is True
    attribution = contract["mandatory_round_attribution"]
    assert set(attribution) >= {
        "pnl_by_actual_model",
        "pnl_by_requested_route",
        "fallback_share",
        "expert_training_support",
        "route_distribution",
        "missingness_by_route",
        "provider_health_by_route",
    }
    assert attribution["pnl_by_actual_model"]["dimensions"] == [
        "expert",
        "fallback",
    ]
    assert attribution["pnl_by_requested_route"]["dimensions"] == [
        "high_vol",
        "bullish",
        "bearish",
        "low_vol",
    ]
    assert attribution["required_for_every_round"] is True
    assert attribution["route_filtering_allowed"] is False
    assert contract["safety"] == SAFETY


def test_moe_runtime_validation_report_matches_fresh_environment_execution() -> None:
    report = verify_frozen_json(MOE_CONFIG_DIR / "moe_artifact_runtime_validation_report.json")
    graph_path = MOE_CONFIG_DIR / "moe_artifact_graph.json"
    observed = validate_static_moe_artifact_in_fresh_environment(
        graph_path=graph_path,
        expected_graph_sha256=_sha256(graph_path),
        repository_root=REPO_ROOT,
    )

    assert report["mandatory_gate_passed"] is True
    assert report["inputs"]["bundle_hash"] == (
        "30d180b028c83146fafd81c8b81269f51fa567b30bc5ab4d3577dd99c256dcf8"
    )
    assert (
        report["checks"]["verified_child_sha256_count"] == observed["verified_child_sha256_count"]
    )
    assert (
        report["checks"]["available_expert_models_loaded"]
        == observed["available_expert_models_loaded"]
    )
    assert (
        report["checks"]["unavailable_expert_stubs_verified"]
        == observed["unavailable_expert_stubs_verified"]
    )
    assert report["synthetic_results"] == observed["fixture_results"]
    assert observed["runtime_validation_passed"] is True
    assert observed["synthetic_deterministic_prediction_equality"] is True
    assert report["state"] == {
        "fresh_collection_authorized": False,
        "fresh_collection_started": False,
        "fresh_outcomes_opened": False,
    }
    assert report["safety"] == observed["safety"] == SAFETY


def test_moe_matched_global_baseline_behavior_is_fully_frozen() -> None:
    contract = verify_frozen_json(MOE_CONFIG_DIR / "moe_matched_global_baseline_contract.json")
    graph = verify_frozen_json(MOE_CONFIG_DIR / "moe_artifact_graph.json")

    assert (
        contract["artifact"]["sha256"] == graph["artifacts"]["moe_global_fallback.json"]["sha256"]
    )
    for descriptor in (
        contract["artifact"]["source_manifest"],
        contract["features"]["feature_contract"],
        contract["features"]["ordered_feature_names"],
        contract["cost_model"]["contract"],
    ):
        assert _sha256(REPO_ROOT / descriptor["path"]) == descriptor["sha256"]
    assert contract["behavior"]["sizing"] == "unit"
    assert contract["behavior"]["execution_policy"] == "HOLD_TO_SETTLEMENT"
    assert contract["behavior"]["NO_TRADE_handling"] == {
        "condition": "no_decision_has_positive_prediction",
        "unit_net_pnl": 0.0,
        "market_retained_in_all_statistics": True,
    }
    assert contract["bootstrap_participation"] == {
        "included_for_every_frozen_confirmatory_market": True,
        "paired_by_market_id_with_candidate": True,
        "same_resample_indices_as_candidate": True,
        "NO_TRADE_contributes_zero": True,
        "missing_market_drop_allowed": False,
        "route_filter_allowed": False,
        "post_hoc_exclusion_allowed": False,
    }
    assert contract["state"]["fresh_collection_authorized"] is False
    assert contract["safety"] == SAFETY


def test_moe_power_analysis_recomputes_variance_without_changing_gates() -> None:
    analysis = verify_frozen_json(MOE_CONFIG_DIR / "moe_confirmatory_power_analysis.json")
    protocol = verify_frozen_json(MOE_CONFIG_DIR / "moe_confirmatory_protocol.json")
    prediction_path = REPO_ROOT / analysis["inputs"]["development_oof_predictions"]["path"]
    rows = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines()]
    ordered_markets = sorted(
        {
            (int(row["market_start_ts"]), str(row["market_id"]))
            for row in rows
            if row["candidate_id"] == "global_baseline"
        }
    )

    def market_pnl(candidate_id: str) -> list[float]:
        selected = {
            str(row["market_id"]): float(row["target"])
            for row in _selected_rows([row for row in rows if row["candidate_id"] == candidate_id])
        }
        return [selected.get(market_id, 0.0) for _, market_id in ordered_markets]

    moe = market_pnl("mixture_of_experts")
    baseline = market_pnl("global_baseline")
    delta = [
        candidate_value - baseline_value
        for candidate_value, baseline_value in zip(moe, baseline, strict=True)
    ]
    paired = analysis["historical_paired_estimates"]["paired_delta_moe_minus_baseline"]
    assert len(delta) == analysis["inputs"]["development_market_count"] == 73
    assert sum(delta) == pytest.approx(paired["total_unit_net_pnl"])
    assert statistics.mean(delta) == pytest.approx(paired["mean_unit_net_pnl"])
    assert statistics.variance(delta) == pytest.approx(paired["sample_variance"])
    assert paired["expected_round_mean_variance_at_n_40"] == pytest.approx(
        statistics.variance(delta) / 40
    )
    assert analysis["gate_snapshot"]["round_gate_canonical_sha256"] == (
        canonical_json_sha256(protocol["round_gate"])
    )
    assert analysis["gate_snapshot"]["bootstrap_canonical_sha256"] == (
        canonical_json_sha256(protocol["bootstrap"])
    )
    assert analysis["state"]["gates_changed"] is False
    assert analysis["state"]["fresh_collection_authorized"] is False
    assert (
        analysis["interpretation"][
            "power_analysis_may_not_change_thresholds_round_count_or_market_targets"
        ]
        is True
    )
    assert analysis["safety"] == SAFETY


def test_moe_authorization_template_pins_scope_but_grants_zero_authority() -> None:
    template = verify_frozen_json(
        MOE_CONFIG_DIR / "moe_fresh_collection_authorization_template.json"
    )
    graph = verify_frozen_json(MOE_CONFIG_DIR / "moe_artifact_graph.json")
    frozen = template["frozen_inputs"]

    assert frozen["artifact_bundle"]["bundle_hash"] == graph["bundle_hash"]
    assert (
        frozen["router_contract"]["sha256"]
        == graph["artifacts"]["moe_router_contract.json"]["sha256"]
    )
    for route in ("high_vol", "bullish", "bearish", "low_vol"):
        assert (
            frozen["expert_artifacts"][route]["sha256"]
            == graph["artifacts"][f"moe_expert_{route}.json"]["sha256"]
        )
    assert (
        frozen["global_fallback"]["sha256"]
        == graph["artifacts"]["moe_global_fallback.json"]["sha256"]
    )
    assert (
        frozen["feature_contract"]["sha256"]
        == graph["artifacts"]["moe_feature_contract.json"]["sha256"]
    )
    for name in (
        "statistical_protocol",
        "collector_protocol",
        "reporting_contract",
        "runtime_validation_report",
        "baseline_behavior_contract",
        "power_analysis",
    ):
        descriptor = frozen[name]
        assert _sha256(REPO_ROOT / descriptor["path"]) == descriptor["sha256"]
    assert template["proposed_collection_scope"] == {
        "number_of_confirmatory_rounds": 2,
        "attempt_cap_per_round": 59,
        "total_attempt_cap": 118,
        "target_quality_valid_market_count_per_round": 40,
        "total_target_quality_valid_market_count": 80,
        "paired_executable_ask_coverage_minimum": 0.95,
        "outcome_blind_capture_required": True,
        "both_round_capture_manifests_frozen_before_any_outcome_open": True,
        "strictly_later_authorization_boundary_required": True,
        "currently_authorized_attempts": 0,
    }
    assert template["template_usable_as_collection_authorization"] is False
    assert (
        template["activation_requirements"]["runtime_validation_mandatory_gate_must_pass"] is True
    )
    assert frozen["runtime_validation_report"]["mandatory_gate_passed"] is True
    assert template["state"] == {
        "fresh_collection_authorized": False,
        "fresh_collection_started": False,
        "fresh_outcomes_opened": False,
    }
    assert template["safety"] == SAFETY


def test_moe_hardening_resume_preserves_provenance_limit_and_safety() -> None:
    resume = verify_frozen_json(MOE_CONFIG_DIR / "lineage_hardening_resume_record.json")
    report = verify_frozen_json(MOE_CONFIG_DIR / "moe_hardening_report.json")

    assert resume["previous_blocked_record"]["preserved_as_historical_record"] is True
    assert resume["resume_basis"]["recorded_parent_source_commit_now_claimed_reachable"] is (False)
    assert resume["resume_basis"]["provenance_limitation_preserved"] is True
    assert resume["outcome_access"]["new_outcomes_opened"] == 0
    assert report["reconciliation"]["reconciliation_passed"] is True
    assert report["static_artifact"]["fresh_clone_load_passed"] is True
    assert report["fresh_collection_remains_blocked"] is True
    assert report["collection_state"] == {
        "fresh_collection_authorized": False,
        "fresh_collection_started": False,
        "fresh_outcomes_opened": False,
        "new_outcome_access_count": 0,
    }
    assert resume["safety"] == report["safety"] == SAFETY


def test_moe_v1_terminal_record_permanently_forbids_collection() -> None:
    record = verify_frozen_json(MOE_CONFIG_DIR / "BTC-15M-MoE-confirmatory-v1-terminal-record.json")

    assert record["terminalized_at_base_commit"] == MOE_V2_BASE_COMMIT
    assert record["v1_fresh_collection_permanently_forbidden"] is True
    assert record["v1_fresh_outcome_access_permanently_forbidden"] is True
    assert record["v1_candidate_artifacts_preserved_for_audit"] is True
    assert record["v1_artifacts_may_not_be_mutated_into_v2"] is True
    assert set(record["terminal_reason_codes"]) == {
        "matched_baseline_information_budget_mismatch",
        "confirmatory_protocol_underpowered_at_observed_development_effect",
        "phase_0_provenance_failure_not_resolved",
        "internal_resume_record_not_independently_auditable",
        "runtime_fixture_coverage_incomplete",
        "full_suite_not_clean_or_baseline_compared",
    }
    assert record["collection_state"] == {
        "fresh_collection_authorized": False,
        "fresh_collection_started": False,
        "fresh_outcomes_opened": False,
    }
    assert record["safety"] == MOE_V2_SAFETY


def test_moe_v1_preterminal_artifacts_remain_byte_identical() -> None:
    record = verify_frozen_json(MOE_CONFIG_DIR / "BTC-15M-MoE-confirmatory-v1-terminal-record.json")

    assert record["artifact_inventory_count"] == 44
    for artifact in record["artifact_inventory"]:
        path = REPO_ROOT / artifact["path"]
        assert path.is_file()
        assert _sha256(path) == artifact["sha256"]
        assert path.stat().st_size == artifact["size_bytes"]


def test_moe_v2_lineage_boundary_preserves_failed_provenance() -> None:
    genesis = verify_frozen_json(MOE_V2_CONFIG_DIR / "lineage_genesis_decision.json")
    manifest = verify_frozen_json(MOE_V2_CONFIG_DIR / "lineage_manifest.json")

    assert genesis["original_recorded_source_commit"] == (
        "364fd65b08849cb36227a3c4bb1b55a62cc68825"
    )
    assert genesis["original_recorded_source_commit_reachable"] is False
    assert genesis["reachable_evaluator_commit"] == ("364fd65afa4908170dbc2ae5ff4f71c8a2475573")
    assert genesis["exact_identity_proven"] is False
    assert genesis["new_lineage_created_instead_of_overriding_failed_v1"] is True
    assert genesis["parent_evidence_role"] == "hypothesis_generation_only"
    assert genesis["parent_evidence_is_not_fresh_validation"] is True
    assert genesis["no_parent_gate_was_relaxed"] is True
    assert genesis["approver"] is None
    assert genesis["request_url"] is None
    assert manifest["lineage_id"] == "BTC-15M-MoE-confirmatory-v2"


def test_moe_v2_baseline_retrained_on_all_113_after_round_selection() -> None:
    manifest = verify_frozen_json(MOE_V2_CONFIG_DIR / "moe_model_manifest.json")
    population = json.loads(
        (REPO_ROOT / manifest["training_population"]["path"]).read_text(encoding="utf-8")
    )

    assert manifest["development_market_count"] == 113
    assert manifest["round_selection_train_market_count"] == 93
    assert manifest["round_selection_validation_market_count"] == 20
    assert manifest["round_selection_model_discarded"] is True
    assert manifest["final_global_baseline_training_market_count"] == 113
    assert manifest["final_global_baseline_uses_validation_labels"] is True
    assert population["round_selection_split_used_only_for_boosting_round_selection"] is True
    assert population["round_selection_fitted_model_discarded"] is True
    assert population["final_global_baseline_training_market_count"] == 113
    assert population["final_global_baseline_uses_validation_labels"] is True
    assert manifest["candidate_and_baseline_information_budget_matched"] is True


def test_moe_v2_available_experts_share_baseline_boosting_round_count() -> None:
    manifest = verify_frozen_json(MOE_V2_CONFIG_DIR / "moe_model_manifest.json")
    selected = manifest["selected_num_boost_round"]

    assert selected == 104
    assert manifest["global_fallback"]["num_boost_round"] == selected
    for route, expert in manifest["experts"].items():
        if expert["available"]:
            assert expert["num_boost_round"] == selected, route
    assert manifest["experts"]["low_vol"]["training_market_count"] == 18
    assert manifest["experts"]["low_vol"]["available"] is False


def test_moe_v2_fallback_is_byte_identical_to_matched_baseline() -> None:
    manifest = verify_frozen_json(MOE_V2_CONFIG_DIR / "moe_model_manifest.json")
    contract = verify_frozen_json(MOE_V2_CONFIG_DIR / "moe_matched_global_baseline_contract.json")

    assert manifest["global_fallback"]["path"] == (manifest["matched_global_baseline"]["path"])
    assert manifest["global_fallback"]["sha256"] == (manifest["matched_global_baseline"]["sha256"])
    assert contract["artifact"]["sha256"] == (manifest["global_fallback"]["sha256"])
    assert contract["artifact"]["byte_identical_to_candidate_global_fallback"] is True
    assert contract["information_budget"]["final_training_market_count"] == 113
    assert contract["cost_and_behavior"]["NO_TRADE_unit_net_pnl"] == 0.0


def test_moe_v2_content_addressed_graph_reconciles() -> None:
    graph = verify_frozen_json(MOE_V2_CONFIG_DIR / "moe_artifact_graph.json")
    manifest = verify_frozen_json(MOE_V2_CONFIG_DIR / "moe_model_manifest.json")
    bundle_dir = REPO_ROOT / graph["bundle_repo_path"]

    assert graph["bundle_hash"] != V1_BUNDLE_HASH
    assert graph["bundle_hash"] == manifest["bundle_hash"]
    assert graph["bundle_hash"] == bundle_dir.name
    assert (bundle_dir / "moe_artifact_graph.json").is_file()
    assert canonical_json_sha256(graph["artifacts"]) == graph["graph_content_sha256"]
    primary_hashes = {
        name: descriptor["sha256"]
        for name, descriptor in graph["artifacts"].items()
        if name != "moe_model_manifest.json"
    }
    assert canonical_json_sha256(primary_hashes) == graph["bundle_hash"]
    for name, descriptor in graph["artifacts"].items():
        path = REPO_ROOT / descriptor["path"]
        assert path.parent == bundle_dir
        assert path.name == name
        assert _sha256(path) == descriptor["sha256"]


def test_moe_v2_fresh_clone_runtime_validation_passes() -> None:
    graph_path = MOE_V2_CONFIG_DIR / "moe_artifact_graph.json"
    first = validate_v2_artifact_in_fresh_environment(
        graph_path=graph_path,
        expected_graph_sha256=_sha256(graph_path),
        repository_root=REPO_ROOT,
    )
    second = load_and_verify_v2_artifact(
        graph_path=graph_path,
        expected_graph_sha256=_sha256(graph_path),
        repository_root=REPO_ROOT,
    )

    assert first["mandatory_gate_passed"] is True
    assert first["all_available_experts_executed"] is True
    assert first["fallback_executed"] is True
    assert first["artifact_hash_mismatch_rejection_passed"] is True
    assert second["fallback_loaded"] is True
    assert second["fresh_collection_authorized"] is False


@pytest.mark.parametrize(
    ("required_path", "route", "actual_model"),
    [
        ("high_vol_native_expert", "high_vol", "moe_expert_high_vol"),
        ("bullish_native_expert", "bullish", "moe_expert_bullish"),
        ("bearish_native_expert", "bearish", "moe_expert_bearish"),
        ("low_vol_global_fallback", "low_vol", "global_baseline_fallback"),
    ],
)
def test_moe_v2_each_material_model_path_executes(
    required_path: str,
    route: str,
    actual_model: str,
) -> None:
    result = _moe_v2_runtime_prediction(required_path)

    assert result["route"] == route
    assert result["actual_model_used"] == actual_model
    assert len(result["raw_probabilities"]) == 2


def test_moe_v2_up_and_down_selection_paths_execute() -> None:
    assert _moe_v2_runtime_prediction("UP_selection")["selected_side"] == "UP"
    assert _moe_v2_runtime_prediction("DOWN_selection")["selected_side"] == "DOWN"


def test_moe_v2_no_trade_and_exact_threshold_reject() -> None:
    no_trade = _moe_v2_runtime_prediction("NO_TRADE")
    boundary = _moe_v2_runtime_prediction("threshold_boundary")

    assert no_trade["accepted"] is False
    assert no_trade["selected_side"] is None
    assert max(no_trade["scores"]) < 0.0
    assert boundary["accepted"] is False
    assert boundary["selected_side"] is None
    assert max(boundary["scores"]) == 0.0


def test_moe_v2_up_tie_break_and_asymmetric_pair_are_deterministic() -> None:
    tie = _moe_v2_runtime_prediction("UP_tie_break")
    asymmetric = _moe_v2_runtime_prediction("asymmetric_pair")

    assert tie["scores"][0] == tie["scores"][1]
    assert tie["selected_side"] == "UP"
    assert asymmetric["raw_probabilities"][0] != asymmetric["raw_probabilities"][1]


def test_moe_v2_router_precedence_and_causality_failures_are_frozen() -> None:
    report = verify_frozen_json(MOE_V2_CONFIG_DIR / "moe_artifact_runtime_validation_report.json")
    router = {row["fixture_id"]: row for row in report["checks"]["router_results"]}
    rejected = {row["fixture_id"]: row for row in report["checks"]["rejection_results"]}

    assert router["high_vol_precedence_over_bullish"]["observed_route"] == "high_vol"
    assert router["bullish_router_path"]["observed_route"] == "bullish"
    assert router["bearish_router_path"]["observed_route"] == "bearish"
    assert router["sideways_low_vol_router_path"]["observed_route"] == "low_vol"
    for fixture_id in (
        "missing_required_router_field",
        "forbidden_future_outcome_router_field",
        "available_after_decision",
        "max_input_after_decision",
        "unknown_route",
    ):
        assert rejected[fixture_id]["rejected"] is True


def test_moe_v2_primary_and_absolute_power_recompute() -> None:
    analysis = verify_frozen_json(MOE_V2_CONFIG_DIR / "moe_confirmatory_power_analysis.json")
    rows = [
        json.loads(line)
        for line in (MOE_V2_CONFIG_DIR / "development_paired_planning_rows.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    delta = [float(row["paired_delta_unit_net_pnl"]) for row in rows]
    absolute = [float(row["moe_unit_net_pnl"]) for row in rows]
    observed = analysis["observed_development_distribution"]

    assert len(rows) == 73
    assert statistics.mean(delta) == pytest.approx(observed["primary_delta"]["mean"])
    assert statistics.variance(delta) == pytest.approx(observed["primary_delta"]["sample_variance"])
    assert statistics.mean(absolute) == pytest.approx(observed["absolute_moe"]["mean"])
    assert statistics.variance(absolute) == pytest.approx(
        observed["absolute_moe"]["sample_variance"]
    )
    z_value = NormalDist().inv_cdf(0.975)
    for row in analysis["design_selection_rows"]:
        primary = NormalDist().cdf(
            row["assumed_primary_delta_mean"]
            * math.sqrt(row["sample_size"])
            / math.sqrt(row["assumed_primary_delta_variance"])
            - z_value
        )
        absolute_probability = NormalDist().cdf(
            row["assumed_absolute_moe_mean"]
            * math.sqrt(row["sample_size"])
            / math.sqrt(row["assumed_absolute_moe_variance"])
            - z_value
        )
        assert primary == pytest.approx(row["estimated_probability_primary_delta_LCB_gt_0"])
        assert absolute_probability == pytest.approx(
            row["estimated_probability_absolute_moe_LCB_gt_0"]
        )


def test_moe_v2_selected_market_count_follows_power_rule() -> None:
    analysis = verify_frozen_json(MOE_V2_CONFIG_DIR / "moe_confirmatory_power_analysis.json")
    eligible = [
        row["sample_size"]
        for row in analysis["design_selection_rows"]
        if row["estimated_probability_primary_delta_LCB_gt_0"] >= 0.8
        and row["estimated_probability_absolute_moe_LCB_gt_0"] >= 0.8
    ]

    assert analysis["candidate_fixed_window_sizes"] == list(MOE_V2_SAMPLE_SIZES)
    assert min(eligible) == 800
    assert analysis["selected_confirmatory_market_count"] == 800
    assert analysis["confirmatory_design_ready"] is True


def test_moe_v2_attempt_cap_follows_conservative_rate_rule() -> None:
    analysis = verify_frozen_json(MOE_V2_CONFIG_DIR / "collection_quality_rate_analysis.json")

    assert analysis["outcomes_labels_or_pnl_read_for_cap_selection"] is False
    assert analysis["model_outputs_used_for_cap_selection"] is False
    assert analysis["attempted_market_count"] == 120
    assert analysis["quality_valid_market_count"] == 113
    assert analysis["attempt_cap"] == math.ceil(
        analysis["target_quality_valid_market_count"]
        / analysis["conservative_quality_rate_lower_bound"]
    )
    assert analysis["attempt_cap"] == 905
    assert analysis["target_is_mathematically_reachable_under_cap"] is True


def test_moe_v2_future_reporting_cannot_drop_or_zero_fill_markets() -> None:
    contract = verify_frozen_json(
        MOE_V2_CONFIG_DIR / "moe_future_evaluation_reporting_contract.json"
    )

    assert contract["population"]["one_row_per_frozen_market_required"] is True
    assert contract["population"]["NO_TRADE_rows_required"] is True
    assert contract["population"]["dropped_market_count_must_equal"] == 0
    assert (
        contract["missingness_semantics"]["missing_value_encoded_as_numeric_zero_allowed"] is False
    )
    assert set(contract["required_panels"]) >= {
        "overall",
        "requested_route",
        "actual_model",
        "expert_vs_fallback",
        "UP_vs_DOWN",
        "regime",
        "provider_health",
        "complete_feature_vs_missing_feature",
        "chronological_half",
        "largest_winner_attribution",
    }
    assert all(contract["forbidden_reporting_behavior"].values())


def test_moe_v2_authorization_template_grants_zero_authority() -> None:
    template = verify_frozen_json(
        MOE_V2_CONFIG_DIR / "moe_fresh_collection_authorization_template.json"
    )

    assert template["template_usable_as_collection_authorization"] is False
    assert template["activation_placeholders"]["authorization_artifact_id"] is None
    assert template["activation_placeholders"]["authorized_by"] is None
    assert template["activation_placeholders"]["authorized_at"] is None
    assert template["activation_placeholders"]["authorization_source_url"] is None
    assert template["activation_placeholders"]["authorization_source_id"] is None
    assert template["activation_placeholders"]["explicit_request_received"] is False
    assert template["activation_placeholders"]["maximum_attempts"] == 905
    assert template["activation_placeholders"]["maximum_markets"] == 800
    assert template["state"] == {
        "fresh_collection_authorized": False,
        "fresh_collection_started": False,
        "fresh_outcomes_opened": False,
    }


def test_moe_v2_regression_ledger_proves_no_new_full_suite_failure() -> None:
    ledger = verify_frozen_json(MOE_V2_CONFIG_DIR / "regression_failure_ledger.json")

    assert ledger["base_commit"] == MOE_V2_BASE_COMMIT
    assert ledger["head_commit"] == ("763a88b49097be41dc818f73db9b225d3dbc7ba2")
    assert ledger["base_pytest"]["failure_count"] == 23
    assert ledger["head_pytest"]["failure_count"] == 23
    assert ledger["pytest_reconciliation"]["added_failure_node_ids"] == []
    assert ledger["pytest_reconciliation"]["changed_message_failure_node_ids"] == []
    assert ledger["pytest_reconciliation"]["new_test_failure_count"] == 0
    assert ledger["pytest_reconciliation"]["head_failures_subset_of_base_failures"] is True
    assert ledger["ruff_reconciliation"]["new_ruff_error_count"] == 0
    assert ledger["base_ruff_errors"] == ledger["head_ruff_errors"] == []
    assert ledger["required_condition_passed"] is True


def test_moe_v2_r1_vendored_health_snapshot_reconciles_and_is_outcome_blind() -> None:
    snapshot_path = MOE_V2_CONFIG_DIR / "collection_attempt_health_snapshot.jsonl"
    manifest_path = MOE_V2_CONFIG_DIR / "collection_attempt_health_manifest.json"
    rows = validate_health_snapshot(
        snapshot_path=snapshot_path,
        manifest_path=manifest_path,
    )
    manifest = verify_frozen_json(manifest_path)

    assert len(rows) == manifest["snapshot_row_count"] == 120
    assert sum(bool(row["quality_valid"]) for row in rows) == 113
    assert manifest["attempted_market_count"] == 120
    assert manifest["quality_valid_market_count"] == 113
    assert manifest["outcomes_labels_or_pnl_read"] is False
    assert manifest["model_outputs_read"] is False
    assert all(not (FORBIDDEN_HEALTH_SNAPSHOT_FIELDS & set(row)) for row in rows)
    derivation_path = REPO_ROOT / manifest["snapshot_derivation_code_path"]
    assert _sha256(derivation_path) == manifest["snapshot_derivation_code_sha256"]


def test_moe_v2_r1_wilson_and_minimum_completion_cap_recompute() -> None:
    analysis = verify_frozen_json(MOE_V2_CONFIG_DIR / "collection_quality_rate_analysis_r1.json")
    lower = wilson_lower_bound(
        success_count=113,
        attempt_count=120,
        confidence=0.975,
    )
    cap, at_cap, before_cap = minimum_attempt_cap(
        target_quality_valid_market_count=800,
        conservative_quality_rate_lower_bound=lower,
        required_completion_probability=0.975,
    )

    assert lower == analysis["conservative_quality_rate_lower_bound"]
    assert analysis["attempt_cap_method"] == ("wilson_lower_bound_plus_binomial_tail_quantile")
    assert cap == analysis["attempt_cap"]
    assert at_cap == analysis["completion_probability_at_attempt_cap"]
    assert before_cap == analysis["completion_probability_at_attempt_cap_minus_one"]
    assert at_cap >= 0.975
    assert before_cap < 0.975
    assert cap >= 800
    assert analysis["attempt_cap_is_minimal"] is True
    assert (
        binomial_tail_probability(
            attempt_count=cap - 1,
            required_success_count=800,
            success_probability=lower,
        )
        < 0.975
    )


def test_moe_v2_r1_exact_window_stops_at_800_and_excludes_market_801() -> None:
    attempts = _moe_v2_r1_attempts(801)
    selected = deterministic_exact_window(attempts)

    assert len(selected) == 800
    assert selected == attempts[:800]
    assert deterministic_exact_window(list(reversed(attempts))) == attempts[:800]
    assert attempts[800] not in selected
    validate_exact_window(
        attempts=attempts,
        selected_markets=selected,
    )


def test_moe_v2_r1_window_rejects_duplicates_skips_replacements_and_model_inputs() -> None:
    attempts = _moe_v2_r1_attempts(801)
    duplicate = [dict(row) for row in attempts]
    duplicate[800]["market_id"] = duplicate[0]["market_id"]
    with pytest.raises(ValueError, match="duplicate"):
        deterministic_exact_window(duplicate)

    expected = deterministic_exact_window(attempts)
    skipped = expected[1:] + [attempts[800]]
    with pytest.raises(ValueError, match="skipped, replaced"):
        validate_exact_window(attempts=attempts, selected_markets=skipped)

    for field, value in (
        ("router_route", "high_vol"),
        ("model_prediction", 0.75),
    ):
        model_controlled = [dict(row) for row in attempts]
        model_controlled[0][field] = value
        with pytest.raises(ValueError, match="forbidden collection-control"):
            deterministic_exact_window(model_controlled)


def test_moe_v2_r1_outcome_access_requires_frozen_exact_manifest() -> None:
    ordered = [f"market-{index:04d}" for index in range(800)]
    manifest = {
        "exact_market_count": 800,
        "ordered_market_ids": ordered,
        "ordered_market_ids_sha256": canonical_json_sha256(ordered),
        "capture_manifest_frozen": True,
        "decision_artifacts_frozen": True,
        "all_artifact_hashes_reconcile": True,
        "all_decisions_frozen": True,
    }

    with pytest.raises(ValueError, match="manifest is required"):
        assert_outcome_access_allowed(capture_manifest=None)
    with pytest.raises(ValueError, match="partial"):
        assert_outcome_access_allowed(
            capture_manifest=manifest,
            requested_market_ids=ordered[:799],
        )
    assert_outcome_access_allowed(
        capture_manifest=manifest,
        requested_market_ids=ordered,
    )


@pytest.mark.parametrize("count", [799, 801])
def test_moe_v2_r1_population_count_other_than_800_fails(count: int) -> None:
    reconciliation = {
        "frozen_window_market_count": count,
        "reported_market_count": count,
        "candidate_market_row_count": count,
        "baseline_market_row_count": count,
        "paired_delta_market_row_count": count,
        "dropped_market_count": 0,
        "duplicate_market_count": 0,
        "out_of_window_market_count": 0,
    }
    with pytest.raises(ValueError, match="exactly 800"):
        validate_population_reconciliation(reconciliation)


def test_moe_v2_r1_exact_population_reconciliation_passes() -> None:
    protocol = verify_frozen_json(MOE_V2_CONFIG_DIR / "moe_confirmatory_protocol_r1.json")
    validate_population_reconciliation(protocol["population_reconciliation"])
    assert protocol["gates"]["quality_valid_market_count"] == {
        "operator": "eq",
        "value": 800,
    }


def test_moe_v2_r1_attempt_hash_chain_tampering_fails() -> None:
    rows = []
    previous = "0" * 64
    for index in range(1, 4):
        row = {
            "attempt_index": index,
            "market_id": f"market-{index}",
            "quality_valid": True,
            "previous_entry_sha256": previous,
        }
        row["entry_sha256"] = canonical_json_sha256(row)
        previous = row["entry_sha256"]
        rows.append(row)
    validate_attempt_hash_chain(rows)

    tampered = [dict(row) for row in rows]
    tampered[1]["quality_valid"] = False
    with pytest.raises(ValueError, match="entry hash mismatch"):
        validate_attempt_hash_chain(tampered)


def test_moe_v2_r1_health_snapshot_raw_evidence_tamper_fails(
    tmp_path: Path,
) -> None:
    source_snapshot = MOE_V2_CONFIG_DIR / "collection_attempt_health_snapshot.jsonl"
    source_manifest = MOE_V2_CONFIG_DIR / "collection_attempt_health_manifest.json"
    snapshot = tmp_path / source_snapshot.name
    manifest = tmp_path / source_manifest.name
    shutil.copy2(source_snapshot, snapshot)
    shutil.copy2(source_manifest, manifest)
    shutil.copy2(source_manifest.with_suffix(".sha256"), manifest.with_suffix(".sha256"))
    rows = snapshot.read_text(encoding="utf-8").splitlines()
    first = json.loads(rows[0])
    first["raw_evidence_manifest_sha256"] = "0" * 64
    rows[0] = json.dumps(first, sort_keys=True, separators=(",", ":"))
    snapshot.write_text("\n".join(rows) + "\n", encoding="utf-8")
    snapshot.with_suffix(".sha256").write_text(
        _sha256(snapshot) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="content SHA-256 mismatch"):
        validate_health_snapshot(
            snapshot_path=snapshot,
            manifest_path=manifest,
        )


def test_moe_v2_r1_raw_evidence_manifest_hash_mismatch_fails(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "raw_evidence_manifest.json"
    manifest.write_text('{"outcome_blind":true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="raw evidence manifest SHA-256 mismatch"):
        verify_raw_evidence_manifest_hash(
            manifest_path=manifest,
            expected_sha256="0" * 64,
        )
    assert verify_raw_evidence_manifest_hash(
        manifest_path=manifest,
        expected_sha256=_sha256(manifest),
    ) == {"outcome_blind": True}


def test_moe_v2_r1_revision_supersedes_old_905_package_without_rewrite() -> None:
    revision = verify_frozen_json(MOE_V2_CONFIG_DIR / "precollection_protocol_revision_record.json")
    old_names = {
        "collection_quality_rate_analysis.json",
        "moe_confirmatory_collector_protocol.json",
        "moe_confirmatory_protocol.json",
        "moe_fresh_collection_authorization_template.json",
    }

    assert revision["candidate_bundle_unchanged"] is True
    assert revision["model_retraining_performed"] is False
    assert revision["supersession"]["superseded_before_authorization"] is True
    assert revision["supersession"]["old_package_may_not_authorize_collection"] is True
    assert set(revision["preserved_original_artifacts"]) == old_names
    for name, descriptor in revision["preserved_original_artifacts"].items():
        assert descriptor["path"].endswith(name)
        assert _sha256(REPO_ROOT / descriptor["path"]) == descriptor["sha256"]


def test_moe_v2_r1_authorization_grants_zero_authority_and_pins_new_cap() -> None:
    template = verify_frozen_json(
        MOE_V2_CONFIG_DIR / "moe_fresh_collection_authorization_template_r1.json"
    )
    analysis = verify_frozen_json(MOE_V2_CONFIG_DIR / "collection_quality_rate_analysis_r1.json")

    assert template["template_usable_as_collection_authorization"] is False
    assert template["old_905_attempt_template_non_authoritative"] is True
    assert template["activation_placeholders"]["explicit_request_received"] is False
    assert template["activation_placeholders"]["maximum_attempts"] == analysis["attempt_cap"]
    assert template["activation_placeholders"]["maximum_markets"] == 800
    assert template["state"] == {
        "fresh_collection_authorized": False,
        "fresh_collection_started": False,
        "fresh_outcomes_opened": False,
    }


def test_moe_v2_r1_candidate_bytes_and_800_target_remain_unchanged() -> None:
    graph = verify_frozen_json(MOE_V2_CONFIG_DIR / "moe_artifact_graph.json")
    collector = verify_frozen_json(
        MOE_V2_CONFIG_DIR / "moe_confirmatory_collector_protocol_r1.json"
    )
    power = verify_frozen_json(MOE_V2_CONFIG_DIR / "moe_confirmatory_power_interpretation_r1.json")

    assert graph["bundle_hash"] == MOE_V2_R1_BUNDLE_HASH
    for descriptor in graph["artifacts"].values():
        assert _sha256(REPO_ROOT / descriptor["path"]) == descriptor["sha256"]
    assert collector["population"]["target_quality_valid_market_count"] == 800
    assert collector["population"]["required_final_quality_valid_market_count"] == 800
    assert power["selected_confirmatory_market_count"] == 800
    assert power["selected_target_unchanged"] is True


def test_moe_v2_r1_power_interpretation_and_empirical_result_reproduce() -> None:
    report = verify_frozen_json(MOE_V2_CONFIG_DIR / "moe_confirmatory_power_interpretation_r1.json")
    rows = [
        json.loads(line)
        for line in (MOE_V2_CONFIG_DIR / "development_paired_planning_rows.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    recomputed = empirical_bootstrap_lcb_crossing_power(rows)

    assert report["primary_and_absolute_lcb_design_ready"] is True
    assert report["overall_all_gate_success_probability_estimated"] is False
    assert report["overall_all_gate_success_probability_not_guaranteed"] is True
    assert report["winner_selection_bias_possible"] is True
    assert report["development_effect_may_be_optimistic"] is True
    assert report["design_criterion"]["variance_multiplier"] == 1.25
    assert report["design_criterion"]["selected_sample_size"] == 800
    assert {
        row["scenario_name"] for row in report["report_only_75pct_and_50pct_effect_power_at_n800"]
    } == {"75pct_of_observed_effect", "50pct_of_observed_effect"}
    assert recomputed == report["empirical_paired_market_bootstrap_validation_at_n800"]


def test_moe_v2_r1_final_regression_ledger_covers_executable_head() -> None:
    ledger_path = MOE_V2_CONFIG_DIR / "regression_failure_ledger_r1.json"
    if not ledger_path.exists():
        pytest.skip("Commit B attestation is created after executable Commit A")
    ledger = verify_frozen_json(ledger_path)
    executable_head = ledger["executable_head_commit"]

    subprocess.run(
        ["git", "cat-file", "-e", f"{executable_head}^{{commit}}"],
        cwd=REPO_ROOT,
        check=True,
    )
    attestation_commit = subprocess.check_output(
        [
            "git",
            "log",
            "--diff-filter=A",
            "--format=%H",
            "-1",
            "--",
            str(ledger_path.relative_to(REPO_ROOT)),
        ],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    changed_in_r1_attestation = set(
        subprocess.check_output(
            [
                "git",
                "diff",
                "--name-only",
                f"{executable_head}..{attestation_commit}",
            ],
            cwd=REPO_ROOT,
            text=True,
        ).splitlines()
    )
    allowed = {
        (
            "examples/v8/polymarket_configs/"
            "BTC-15M-MoE-confirmatory-v2/regression_failure_ledger_r1.json"
        ),
        (
            "examples/v8/polymarket_configs/"
            "BTC-15M-MoE-confirmatory-v2/regression_failure_ledger_r1.sha256"
        ),
        (
            "examples/v8/polymarket_configs/"
            "BTC-15M-MoE-confirmatory-v2/final_precollection_hardening_report.json"
        ),
        (
            "examples/v8/polymarket_configs/"
            "BTC-15M-MoE-confirmatory-v2/final_precollection_hardening_report.sha256"
        ),
    }
    assert ledger["base_commit"] == MOE_V2_R1_BASE_COMMIT
    assert ledger["attestation_commit"] is None
    assert changed_in_r1_attestation <= allowed
    assert ledger["pytest_reconciliation"]["new_test_failure_count"] == 0
    assert ledger["pytest_reconciliation"]["head_failures_subset_of_base_failures"] is True
    assert ledger["ruff_reconciliation"]["new_ruff_error_count"] == 0
    assert ledger["required_condition_passed"] is True


def test_moe_v2_r2_test_fixture_requires_explicit_test_opt_in() -> None:
    with pytest.raises(ValueError, match="test fixture cannot authorize"):
        validate_manual_collection_authorization(
            replace(_moe_v2_r2_authorization(), allow_test_fixture=False),
            repository_root=REPO_ROOT,
        )
    result = validate_manual_collection_authorization(
        _moe_v2_r2_authorization(),
        repository_root=REPO_ROOT,
    )

    assert result["authorization_validated"] is True
    assert result["capture_only_authorized"] is False
    assert result["test_fixture_only"] is True
    assert all(value is False for value in result["permissions"].values())


def test_moe_v2_r2_valid_production_shaped_authorization_passes(
    tmp_path: Path,
) -> None:
    frozen = frozen_authorization_artifacts(REPO_ROOT)
    payload = build_manual_authorization_payload(
        repository_root=REPO_ROOT,
        authorization_artifact_id="manual-contract-shape-test-only",
        authorization_request_text_sha256="3" * 64,
        authorization_decision_text_sha256="4" * 64,
        approver_identity="test-human-approver",
        authorization_source_id="test-source-id",
        authorization_source_url="https://example.invalid/manual-test",
        authorization_timestamp="2029-12-31T23:59:00+00:00",
        strictly_later_than_timestamp="2030-01-01T00:00:00+00:00",
    )
    path = tmp_path / "production-shaped-authorization.json"
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    expectation = AuthorizationExpectation(
        artifact_path=path,
        expected_artifact_sha256=_sha256(path),
        authorization_request_text_sha256="3" * 64,
        authorization_decision_text_sha256="4" * 64,
        approver_identity="test-human-approver",
        authorization_source_id="test-source-id",
        authorization_source_url="https://example.invalid/manual-test",
        authorization_timestamp="2029-12-31T23:59:00+00:00",
        strictly_later_than_timestamp="2030-01-01T00:00:00+00:00",
        expected_frozen_artifact_sha256={
            key: descriptor["sha256"] for key, descriptor in frozen.items()
        },
    )

    result = validate_manual_collection_authorization(
        expectation,
        repository_root=REPO_ROOT,
    )
    assert result["authorization_validated"] is True
    assert result["capture_only_authorized"] is True
    assert result["test_fixture_only"] is False
    assert result["permissions"]["outcome_access_enabled"] is False


def test_moe_v2_r2_inactive_manual_authorization_template_fails_closed() -> None:
    template_path = MOE_V2_CONFIG_DIR / "moe_manual_collection_authorization_template_r2.json"
    template = verify_frozen_json(template_path)

    assert template["template_usable_as_collection_authorization"] is False
    assert template["explicit_manual_authorization"] is False
    assert template["outcome_blind_capture_authorized"] is False
    assert template["authorization_artifact_id"] is None
    assert template["state"] == MOE_V2_R2_STATE_BLOCKED
    with pytest.raises(ValueError, match="authorization field mismatch"):
        validate_manual_collection_authorization(
            replace(
                _moe_v2_r2_authorization(),
                artifact_path=template_path,
                expected_artifact_sha256=_sha256(template_path),
                allow_test_fixture=False,
            ),
            repository_root=REPO_ROOT,
        )


def test_moe_v2_r2_authorization_missing_approver_fails(
    tmp_path: Path,
) -> None:
    payload = _moe_v2_r2_authorization_payload()
    del payload["approver_identity"]
    expectation = _moe_v2_r2_mutated_authorization(tmp_path, payload)

    with pytest.raises(ValueError, match="missing required fields"):
        validate_manual_collection_authorization(
            expectation,
            repository_root=REPO_ROOT,
        )


def test_moe_v2_r2_authorization_missing_request_hash_fails(
    tmp_path: Path,
) -> None:
    payload = _moe_v2_r2_authorization_payload()
    del payload["authorization_request_text_sha256"]
    expectation = _moe_v2_r2_mutated_authorization(tmp_path, payload)

    with pytest.raises(ValueError, match="missing required fields"):
        validate_manual_collection_authorization(
            expectation,
            repository_root=REPO_ROOT,
        )


def test_moe_v2_r2_authorization_wrong_frozen_protocol_hash_fails() -> None:
    authorization = _moe_v2_r2_authorization()
    wrong_hashes = dict(authorization.expected_frozen_artifact_sha256)
    wrong_hashes["collector_protocol"] = "0" * 64

    with pytest.raises(ValueError, match="caller frozen protocol hash mismatch"):
        validate_manual_collection_authorization(
            replace(
                authorization,
                expected_frozen_artifact_sha256=wrong_hashes,
            ),
            repository_root=REPO_ROOT,
        )


def test_moe_v2_r2_pre_boundary_market_fails(tmp_path: Path) -> None:
    rows = _moe_v2_r2_source_attempts(1)
    boundary_ms = _moe_v2_r2_boundary_ms()
    rows[0]["market_start_ts"] = boundary_ms - 1
    _moe_v2_r2_rehash_attempts(rows)

    with pytest.raises(ValueError, match="strictly later"):
        _moe_v2_r2_validate_attempt_rows(rows, tmp_path)


def test_moe_v2_r2_market_at_exact_boundary_fails(
    tmp_path: Path,
) -> None:
    rows = _moe_v2_r2_source_attempts(1)
    rows[0]["market_start_ts"] = _moe_v2_r2_boundary_ms()
    _moe_v2_r2_rehash_attempts(rows)

    with pytest.raises(ValueError, match="strictly later"):
        _moe_v2_r2_validate_attempt_rows(rows, tmp_path)


def test_moe_v2_r2_post_boundary_market_passes(tmp_path: Path) -> None:
    result = _moe_v2_r2_validate_attempt_rows(
        _moe_v2_r2_source_attempts(1),
        tmp_path,
    )

    assert result["attempt_count"] == 1
    assert result["quality_valid_count"] == 1
    assert result["outcome_accessed"] is False


def test_moe_v2_r2_non_btc_15m_market_fails(tmp_path: Path) -> None:
    rows = _moe_v2_r2_source_attempts(1)
    rows[0]["market_family"] = "eth_updown_15m"
    _moe_v2_r2_rehash_attempts(rows)

    with pytest.raises(ValueError, match="market family"):
        _moe_v2_r2_validate_attempt_rows(rows, tmp_path)


def test_moe_v2_r2_attempt_927_fails(tmp_path: Path) -> None:
    rows = [_moe_v2_r2_source_attempts(1)[0] for _ in range(927)]

    with pytest.raises(ValueError, match="attempt 927"):
        _moe_v2_r2_validate_attempt_rows(rows, tmp_path)


def test_moe_v2_r2_cap_exhaustion_is_terminal_incomplete_failure(
    moe_v2_r2_repo_temp_dir: Path,
) -> None:
    normalized = _moe_v2_r2_build_normalized_population(
        moe_v2_r2_repo_temp_dir,
        attempt_count=926,
        quality_valid_count=799,
    )
    result = select_exact_authorized_window(
        normalized_attempt_ledger_path=normalized,
        expected_normalized_attempt_ledger_sha256=_sha256(normalized),
        authorization=_moe_v2_r2_authorization(),
        repository_root=REPO_ROOT,
    )

    assert result["window_complete"] is False
    assert result["window_incomplete_failed"] is True
    assert result["failure_reason"] == "attempt_cap_exhausted_before_exact_800"
    assert result["attempts_consumed"] == 926
    assert result["quality_valid_unique_market_count"] == 799
    assert result["extension_allowed"] is False
    assert result["replacement_allowed"] is False
    assert result["reset_allowed"] is False
    assert result["new_attempt_budget_allowed"] is False


def test_moe_v2_r2_exact_800_stops_selection() -> None:
    selection = _moe_v2_r2_selection()

    assert selection["window_complete"] is True
    assert selection["window_stop_attempt_index"] == 800
    assert selection["selected_attempt_indices"] == list(range(1, 801))
    assert len(selection["selected_market_ids"]) == 800


def test_moe_v2_r2_concurrent_market_801_is_audit_only() -> None:
    selection = _moe_v2_r2_selection()

    assert selection["audit_only_post_boundary_attempt_indices"] == [801]
    assert selection["audit_only_post_boundary_rows"][0]["market_id"].endswith("0801")
    assert (
        selection["audit_only_post_boundary_rows"][0]["market_id"]
        not in (selection["selected_market_ids"])
    )


def test_moe_v2_r2_caller_quality_valid_without_evidence_fails(
    tmp_path: Path,
) -> None:
    rows = _moe_v2_r2_source_attempts(1)
    rows[0]["quality_valid"] = True
    _moe_v2_r2_rehash_attempts(rows)

    with pytest.raises(ValueError, match="caller-supplied"):
        _moe_v2_r2_validate_attempt_rows(rows, tmp_path)


def test_moe_v2_r2_raw_evidence_hash_mismatch_fails(
    tmp_path: Path,
) -> None:
    rows = _moe_v2_r2_source_attempts(1)
    rows[0]["raw_evidence_manifest_sha256"] = "0" * 64
    _moe_v2_r2_rehash_attempts(rows)

    with pytest.raises(ValueError, match="hash not found"):
        _moe_v2_r2_validate_attempt_rows(rows, tmp_path)


def test_moe_v2_r2_missing_required_raw_stream_fails(
    tmp_path: Path,
    moe_v2_r2_repo_temp_dir: Path,
) -> None:
    rows = _moe_v2_r2_attempts_with_mutated_first_evidence(
        moe_v2_r2_repo_temp_dir,
        lambda record: record["raw_streams"].pop("raw_polymarket_orderbooks.jsonl"),
    )

    with pytest.raises(ValueError, match="missing required raw stream"):
        _moe_v2_r2_validate_attempt_rows(rows, tmp_path)


def test_moe_v2_r2_raw_evidence_causality_violation_fails(
    tmp_path: Path,
    moe_v2_r2_repo_temp_dir: Path,
) -> None:
    def violate(record: dict) -> None:
        record["causality"]["available_at_ts"] = record["causality"]["decision_ts"] + 1

    rows = _moe_v2_r2_attempts_with_mutated_first_evidence(
        moe_v2_r2_repo_temp_dir,
        violate,
    )

    with pytest.raises(ValueError, match="raw evidence causality violation"):
        _moe_v2_r2_validate_attempt_rows(rows, tmp_path)


def test_moe_v2_r2_attempt_hash_chain_tampering_fails(
    tmp_path: Path,
) -> None:
    rows = _moe_v2_r2_source_attempts(2)
    rows[1]["market_id"] = "tampered-without-rehash"

    with pytest.raises(ValueError, match="hash-chain entry mismatch"):
        _moe_v2_r2_validate_attempt_rows(rows, tmp_path)


def test_moe_v2_r2_duplicate_attempt_id_fails(tmp_path: Path) -> None:
    rows = _moe_v2_r2_source_attempts(2)
    rows[1]["attempt_id"] = rows[0]["attempt_id"]
    _moe_v2_r2_rehash_attempts(rows)

    with pytest.raises(ValueError, match="duplicate attempt ID"):
        _moe_v2_r2_validate_attempt_rows(rows, tmp_path)


def test_moe_v2_r2_skipped_valid_market_fails() -> None:
    selection = _moe_v2_r2_selection()
    skipped = selection["selected_market_ids"][1:] + [
        selection["audit_only_post_boundary_rows"][0]["market_id"]
    ]

    with pytest.raises(ValueError, match="skipped, replaced"):
        validate_exact_window_selection(
            selection=selection,
            supplied_market_ids=skipped,
        )


def test_moe_v2_r2_replacement_market_fails() -> None:
    selection = _moe_v2_r2_selection()
    replacement = list(selection["selected_market_ids"])
    replacement[-1] = "replacement-market"

    with pytest.raises(ValueError, match="skipped, replaced"):
        validate_exact_window_selection(
            selection=selection,
            supplied_market_ids=replacement,
        )


def test_moe_v2_r2_capture_manifest_missing_required_field_fails() -> None:
    manifest = _moe_v2_r2_capture_manifest()
    del manifest["raw_evidence_manifest_set_sha256"]

    with pytest.raises(ValueError, match="missing fields"):
        validate_capture_manifest_structure(manifest)


def test_moe_v2_r2_capture_manifest_sidecar_mismatch_fails(
    tmp_path: Path,
) -> None:
    kwargs = _moe_v2_r2_outcome_kwargs()
    manifest_path = tmp_path / "confirmatory_capture_manifest.json"
    shutil.copy2(kwargs["capture_manifest_path"], manifest_path)
    manifest_path.with_suffix(".sha256").write_text(
        "0" * 64 + "\n",
        encoding="utf-8",
    )
    kwargs["capture_manifest_path"] = manifest_path

    with pytest.raises(ValueError, match="sidecar mismatch"):
        validate_and_authorize_exact_outcome_access(**kwargs)


def test_moe_v2_r2_candidate_decision_hash_mismatch_fails() -> None:
    kwargs = _moe_v2_r2_outcome_kwargs()
    kwargs["expected_candidate_decision_rows_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="candidate decision rows SHA-256"):
        validate_and_authorize_exact_outcome_access(**kwargs)


def test_moe_v2_r2_baseline_decision_hash_mismatch_fails() -> None:
    kwargs = _moe_v2_r2_outcome_kwargs()
    kwargs["expected_baseline_decision_rows_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="baseline decision rows SHA-256"):
        validate_and_authorize_exact_outcome_access(**kwargs)


def test_moe_v2_r2_raw_evidence_set_hash_mismatch_fails(
    tmp_path: Path,
) -> None:
    kwargs = _moe_v2_r2_outcome_kwargs()
    manifest = _moe_v2_r2_capture_manifest()
    manifest["raw_evidence_manifest_set_sha256"] = "0" * 64
    path = _moe_v2_r2_write_frozen_json(
        tmp_path / "confirmatory_capture_manifest.json",
        manifest,
    )
    kwargs["capture_manifest_path"] = path
    kwargs["expected_capture_manifest_sha256"] = _sha256(path)

    with pytest.raises(ValueError, match="raw evidence manifest set hash mismatch"):
        validate_and_authorize_exact_outcome_access(**kwargs)


def test_moe_v2_r2_attempts_consumed_mismatch_fails(
    tmp_path: Path,
) -> None:
    kwargs = _moe_v2_r2_outcome_kwargs()
    manifest = _moe_v2_r2_capture_manifest()
    manifest["attempts_consumed"] = 800
    manifest["unused_attempt_capacity"] = 126
    path = _moe_v2_r2_write_frozen_json(
        tmp_path / "confirmatory_capture_manifest.json",
        manifest,
    )
    kwargs["capture_manifest_path"] = path
    kwargs["expected_capture_manifest_sha256"] = _sha256(path)

    with pytest.raises(ValueError, match="attempts-consumed mismatch"):
        validate_and_authorize_exact_outcome_access(**kwargs)


def test_moe_v2_r2_unused_attempt_capacity_mismatch_fails(
    tmp_path: Path,
) -> None:
    kwargs = _moe_v2_r2_outcome_kwargs()
    manifest = _moe_v2_r2_capture_manifest()
    manifest["unused_attempt_capacity"] += 1
    path = _moe_v2_r2_write_frozen_json(
        tmp_path / "confirmatory_capture_manifest.json",
        manifest,
    )
    kwargs["capture_manifest_path"] = path
    kwargs["expected_capture_manifest_sha256"] = _sha256(path)

    with pytest.raises(ValueError, match="unused attempt capacity mismatch"):
        validate_and_authorize_exact_outcome_access(**kwargs)


def test_moe_v2_r2_requested_market_ids_omitted_fails() -> None:
    kwargs = _moe_v2_r2_outcome_kwargs()
    kwargs["requested_market_ids"] = None

    with pytest.raises(ValueError, match="mandatory"):
        validate_and_authorize_exact_outcome_access(**kwargs)


def test_moe_v2_r2_partial_outcome_opening_fails() -> None:
    kwargs = _moe_v2_r2_outcome_kwargs()
    kwargs["requested_market_ids"] = kwargs["requested_market_ids"][:-1]

    with pytest.raises(ValueError, match="partial, reordered, or incremental"):
        validate_and_authorize_exact_outcome_access(**kwargs)


def test_moe_v2_r2_reordered_outcome_opening_fails() -> None:
    kwargs = _moe_v2_r2_outcome_kwargs()
    requested = list(kwargs["requested_market_ids"])
    requested[0], requested[1] = requested[1], requested[0]
    kwargs["requested_market_ids"] = requested

    with pytest.raises(ValueError, match="partial, reordered, or incremental"):
        validate_and_authorize_exact_outcome_access(**kwargs)


def test_moe_v2_r2_exact_full_population_outcome_gate_passes() -> None:
    result = validate_and_authorize_exact_outcome_access(**_moe_v2_r2_outcome_kwargs())

    assert result["outcome_access_validation_passed"] is True
    assert result["validation_fixture_only"] is True
    assert result["exact_full_population_authorized"] is False
    assert result["gate_would_authorize_if_non_fixture"] is True
    assert result["outcome_opened_by_validator"] is False
    assert result["requested_market_count"] == 800


def test_moe_v2_r2_candidate_bundle_remains_byte_identical() -> None:
    graph = verify_frozen_json(MOE_V2_CONFIG_DIR / "moe_artifact_graph.json")
    fixture = verify_frozen_json(MOE_V2_R2_FIXTURE_DIR / "fixture_manifest.json")

    assert graph["bundle_hash"] == MOE_V2_R2_BUNDLE_HASH
    assert fixture["candidate_bundle_hash"] == MOE_V2_R2_BUNDLE_HASH
    for descriptor in graph["artifacts"].values():
        assert _sha256(REPO_ROOT / descriptor["path"]) == descriptor["sha256"]


def test_moe_v2_r2_target_remains_exactly_800() -> None:
    collector = verify_frozen_json(
        MOE_V2_CONFIG_DIR / "moe_confirmatory_collector_protocol_r2.json"
    )
    manifest = _moe_v2_r2_capture_manifest()

    assert MOE_V2_R2_TARGET == 800
    assert collector["target_quality_valid_market_count"] == 800
    assert manifest["exact_market_count"] == 800


def test_moe_v2_r2_attempt_cap_remains_exactly_926() -> None:
    collector = verify_frozen_json(
        MOE_V2_CONFIG_DIR / "moe_confirmatory_collector_protocol_r2.json"
    )
    analysis = verify_frozen_json(MOE_V2_CONFIG_DIR / "collection_quality_rate_analysis_r1.json")

    assert MOE_V2_R2_ATTEMPT_CAP == 926
    assert collector["maximum_attempts"] == 926
    assert analysis["attempt_cap"] == 926


def test_moe_v2_r2_all_safety_flags_remain_false() -> None:
    fixture = verify_frozen_json(MOE_V2_R2_FIXTURE_DIR / "fixture_manifest.json")
    for name in (
        "moe_confirmatory_collector_protocol_r2.json",
        "moe_exact_outcome_access_protocol_r2.json",
        "moe_manual_collection_authorization_contract_r2.json",
        "moe_manual_collection_authorization_template_r2.json",
        "collection_attempt_health_reason_semantics_r2.json",
    ):
        payload = verify_frozen_json(MOE_V2_CONFIG_DIR / name)
        assert payload["safety"] == MOE_V2_R2_SAFETY
        assert all(value is False for value in payload["safety"].values())
    assert fixture["state"] == MOE_V2_R2_STATE_BLOCKED
    assert fixture["safety"] == MOE_V2_R2_SAFETY
    assert fixture["collection_started"] is False
    assert fixture["outcome_opened"] is False


def test_moe_v2_r2_health_reason_semantics_do_not_invent_common_reasons() -> None:
    amendment = verify_frozen_json(
        MOE_V2_CONFIG_DIR / "collection_attempt_health_reason_semantics_r2.json"
    )

    assert amendment["source_attempted_market_count"] == 120
    assert amendment["source_quality_valid_market_count"] == 113
    assert amendment["invalid_attempt_record_count"] == 7
    assert amendment["old_broad_reason_attachment_is_authoritative"] is False
    assert all(
        record["quality_failure_reason_codes"]
        != [
            "chainlink_causality_failed",
            "decision_row_count_not_2",
            "feature_rows_missing",
            "market_row_coverage_failed",
            "orderbook_full_window_coverage_failed",
            "paired_executable_ask_coverage_failed",
            "provider_orderbook_snapshot_coverage_failed",
        ]
        for record in amendment["invalid_attempt_records"]
    )
    assert all(
        record["quality_failure_reason_codes"]
        or record["quality_failure_reason_codes"] == ["unknown"]
        for record in amendment["invalid_attempt_records"]
    )


def test_moe_v2_r2_final_attestation_only_adds_reports() -> None:
    report_path = MOE_V2_CONFIG_DIR / "final_collection_boundary_hardening_report.json"
    if not report_path.exists():
        pytest.skip("Commit B attestation is created after executable Commit A")
    report = verify_frozen_json(report_path)
    ledger = verify_frozen_json(MOE_V2_CONFIG_DIR / "regression_failure_ledger_r2.json")
    executable_head = report["executable_head_commit"]
    attestation_commit = subprocess.check_output(
        [
            "git",
            "log",
            "--diff-filter=A",
            "--format=%H",
            "-1",
            "--",
            str(report_path.relative_to(REPO_ROOT)),
        ],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    changed = set(
        subprocess.check_output(
            [
                "git",
                "diff",
                "--name-only",
                f"{executable_head}..{attestation_commit}",
            ],
            cwd=REPO_ROOT,
            text=True,
        ).splitlines()
    )
    prefix = "examples/v8/polymarket_configs/BTC-15M-MoE-confirmatory-v2/"
    assert changed == {
        prefix + "regression_failure_ledger_r2.json",
        prefix + "regression_failure_ledger_r2.sha256",
        prefix + "final_collection_boundary_hardening_report.json",
        prefix + "final_collection_boundary_hardening_report.sha256",
    }
    assert ledger["executable_head_commit"] == executable_head
    assert ledger["pytest_reconciliation"]["new_test_failure_count"] == 0
    assert ledger["pytest_reconciliation"]["head_failures_subset_of_base_failures"] is True
    assert ledger["ruff_reconciliation"]["new_ruff_error_count"] == 0
    assert ledger["required_condition_passed"] is True


def test_moe_v2_all_frozen_json_keeps_safety_false() -> None:
    for path in sorted(MOE_V2_CONFIG_DIR.glob("*.json")):
        payload = verify_frozen_json(path)
        if "safety" in payload:
            assert payload["safety"] == MOE_V2_SAFETY, path.name
        state = payload.get("state", {})
        for field in (
            "fresh_collection_authorized",
            "fresh_collection_started",
            "fresh_outcomes_opened",
        ):
            if field in state:
                assert state[field] is False, (path.name, field)


@pytest.fixture
def moe_v2_r2_repo_temp_dir() -> Path:
    with tempfile.TemporaryDirectory(
        prefix=".moe-v2-r2-test-",
        dir=REPO_ROOT,
    ) as directory:
        yield Path(directory)


def _moe_v2_r2_authorization_payload() -> dict:
    return json.loads(
        (MOE_V2_R2_FIXTURE_DIR / "manual_authorization_fixture.json").read_text(encoding="utf-8")
    )


def _moe_v2_r2_authorization() -> AuthorizationExpectation:
    payload = _moe_v2_r2_authorization_payload()
    artifact_path = MOE_V2_R2_FIXTURE_DIR / "manual_authorization_fixture.json"
    return AuthorizationExpectation(
        artifact_path=artifact_path,
        expected_artifact_sha256=_sha256(artifact_path),
        authorization_request_text_sha256=payload["authorization_request_text_sha256"],
        authorization_decision_text_sha256=payload["authorization_decision_text_sha256"],
        approver_identity=payload["approver_identity"],
        authorization_source_id=payload["authorization_source_id"],
        authorization_source_url=payload["authorization_source_url"],
        authorization_timestamp=payload["authorization_timestamp"],
        strictly_later_than_timestamp=payload["strictly_later_than_timestamp"],
        expected_frozen_artifact_sha256={
            key: descriptor["sha256"]
            for key, descriptor in frozen_authorization_artifacts(REPO_ROOT).items()
        },
        allow_test_fixture=True,
    )


def _moe_v2_r2_mutated_authorization(
    tmp_path: Path,
    payload: dict,
) -> AuthorizationExpectation:
    path = tmp_path / "manual_authorization_fixture.json"
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return replace(
        _moe_v2_r2_authorization(),
        artifact_path=path,
        expected_artifact_sha256=_sha256(path),
    )


def _moe_v2_r2_source_attempts(count: int) -> list[dict]:
    lines = (
        (MOE_V2_R2_FIXTURE_DIR / "authorized_attempt_ledger.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    return [json.loads(line) for line in lines[:count]]


def _moe_v2_r2_rehash_attempts(rows: list[dict]) -> None:
    previous = "0" * 64
    for index, row in enumerate(rows, start=1):
        row["attempt_index"] = index
        row["previous_entry_sha256"] = previous
        row.pop("entry_sha256", None)
        row["entry_sha256"] = canonical_json_sha256(row)
        previous = row["entry_sha256"]


def _moe_v2_r2_validate_attempt_rows(
    rows: list[dict],
    tmp_path: Path,
) -> dict:
    source = tmp_path / "authorized_attempt_ledger.jsonl"
    source.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    return validate_authorized_attempt_ledger(
        authorization=_moe_v2_r2_authorization(),
        attempt_ledger_path=source,
        expected_attempt_ledger_sha256=_sha256(source),
        normalized_output_path=tmp_path / "normalized_attempt_ledger.jsonl",
        repository_root=REPO_ROOT,
    )


def _moe_v2_r2_boundary_ms() -> int:
    value = _moe_v2_r2_authorization_payload()["strictly_later_than_timestamp"]
    return int(__import__("datetime").datetime.fromisoformat(value).timestamp() * 1000)


def _moe_v2_r2_selection() -> dict:
    normalized = MOE_V2_R2_FIXTURE_DIR / "normalized_attempt_ledger.jsonl"
    return select_exact_authorized_window(
        normalized_attempt_ledger_path=normalized,
        expected_normalized_attempt_ledger_sha256=_sha256(normalized),
        authorization=_moe_v2_r2_authorization(),
        repository_root=REPO_ROOT,
    )


def _moe_v2_r2_build_normalized_population(
    directory: Path,
    *,
    attempt_count: int,
    quality_valid_count: int,
) -> Path:
    template_records = [
        json.loads(line)
        for line in (MOE_V2_R2_FIXTURE_DIR / "raw_evidence_manifest_index.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    stream_descriptors = template_records[0]["raw_streams"]
    boundary_ms = _moe_v2_r2_boundary_ms()
    records = []
    for index in range(1, attempt_count + 1):
        valid = index <= quality_valid_count
        record = {
            "schema_version": "bigan-synthetic-raw-evidence-record-r2",
            "attempt_id": f"cap-test-attempt-{index:04d}",
            "market_id": f"cap-test-market-{index:04d}",
            "raw_streams": stream_descriptors,
            "paired_executable_ask_capture_attempted": True,
            "missing_values_encoded_as_numeric_zero": False,
            "complement_quote_proxy_used": False,
            "causality": {
                "available_at_ts": boundary_ms + index * 900_000 + 10,
                "decision_ts": boundary_ms + index * 900_000 + 300_000,
                "max_input_ts": boundary_ms + index * 900_000 + 20,
                "feature_cutoff_ts": boundary_ms + index * 900_000 + 30,
            },
            "quality_observations": {
                "market_identity_complete": True,
                "provider_capture_complete": valid,
                "paired_executable_asks_complete": True,
                "book_capture_complete": True,
                "chainlink_capture_complete": True,
            },
            "direct_failure_reason_codes": [],
            "derived_failure_reason_codes": [],
        }
        record["record_sha256"] = canonical_json_sha256(record)
        records.append(record)
    raw_index = directory / "raw_evidence_manifest_index.jsonl"
    raw_index.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in records),
        encoding="utf-8",
    )
    raw_path = raw_index.relative_to(REPO_ROOT).as_posix()
    raw_sha = _sha256(raw_index)
    authorization = _moe_v2_r2_authorization()
    normalized_rows = []
    for index, evidence in enumerate(records, start=1):
        valid = index <= quality_valid_count
        causality = evidence["causality"]
        normalized_rows.append(
            {
                "schema_version": "bigan-btc-15m-authorized-attempt-r2",
                "lineage_id": "BTC-15M-MoE-confirmatory-v2",
                "authorization_artifact_id": ("synthetic-boundary-validation-only"),
                "authorization_artifact_sha256": (authorization.expected_artifact_sha256),
                "authorization_boundary_timestamp": (authorization.strictly_later_than_timestamp),
                "attempt_index": index,
                "attempt_id": evidence["attempt_id"],
                "market_id": evidence["market_id"],
                "market_family": "btc_updown_15m",
                "market_horizon_seconds": 900,
                "market_start_ts": boundary_ms + index * 900_000,
                "decision_ts": causality["decision_ts"],
                "available_at_ts": causality["available_at_ts"],
                "max_input_ts": causality["max_input_ts"],
                "feature_cutoff_ts": causality["feature_cutoff_ts"],
                "paired_executable_ask_capture_attempted": True,
                "missing_values_encoded_as_numeric_zero": False,
                "complement_quote_proxy_used": False,
                "raw_evidence_manifest_index_path": raw_path,
                "raw_evidence_manifest_index_sha256": raw_sha,
                "raw_evidence_manifest_sha256": evidence["record_sha256"],
                "quality_valid": valid,
                "direct_failure_reason_codes": [],
                "derived_failure_reason_codes": [],
                "quality_failure_reason_codes": [] if valid else ["unknown"],
            }
        )
    normalized = directory / "normalized_attempt_ledger.jsonl"
    normalized.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in normalized_rows
        ),
        encoding="utf-8",
    )
    return normalized


def _moe_v2_r2_attempts_with_mutated_first_evidence(
    directory: Path,
    mutate: object,
) -> list[dict]:
    raw_records = [
        json.loads(line)
        for line in (MOE_V2_R2_FIXTURE_DIR / "raw_evidence_manifest_index.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    first = raw_records[0]
    first.pop("record_sha256")
    mutate(first)
    first["record_sha256"] = canonical_json_sha256(first)
    raw_index = directory / "raw_evidence_manifest_index.jsonl"
    raw_index.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in raw_records
        ),
        encoding="utf-8",
    )
    rows = _moe_v2_r2_source_attempts(1)
    rows[0]["raw_evidence_manifest_index_path"] = raw_index.relative_to(REPO_ROOT).as_posix()
    rows[0]["raw_evidence_manifest_index_sha256"] = _sha256(raw_index)
    rows[0]["raw_evidence_manifest_sha256"] = first["record_sha256"]
    _moe_v2_r2_rehash_attempts(rows)
    return rows


def _moe_v2_r2_capture_manifest() -> dict:
    return json.loads(
        (MOE_V2_R2_FIXTURE_DIR / "confirmatory_capture_manifest.json").read_text(encoding="utf-8")
    )


def _moe_v2_r2_outcome_kwargs() -> dict:
    fixture = verify_frozen_json(MOE_V2_R2_FIXTURE_DIR / "fixture_manifest.json")
    manifest = _moe_v2_r2_capture_manifest()

    def artifact(name: str) -> tuple[Path, str]:
        descriptor = fixture[name]
        return REPO_ROOT / descriptor["path"], descriptor["sha256"]

    capture_path, capture_sha = artifact("capture_manifest")
    normalized_path, normalized_sha = artifact("normalized_attempt_ledger")
    candidate_path, candidate_sha = artifact("candidate_decision_rows")
    baseline_path, baseline_sha = artifact("baseline_decision_rows")
    raw_path, raw_sha = artifact("raw_evidence_manifest_index")
    collector = MOE_V2_CONFIG_DIR / "moe_confirmatory_collector_protocol_r2.json"
    statistical = MOE_V2_CONFIG_DIR / "moe_confirmatory_protocol_r1.json"
    return {
        "authorization": _moe_v2_r2_authorization(),
        "capture_manifest_path": capture_path,
        "expected_capture_manifest_sha256": capture_sha,
        "normalized_attempt_ledger_path": normalized_path,
        "expected_normalized_attempt_ledger_sha256": normalized_sha,
        "candidate_decision_rows_path": candidate_path,
        "expected_candidate_decision_rows_sha256": candidate_sha,
        "baseline_decision_rows_path": baseline_path,
        "expected_baseline_decision_rows_sha256": baseline_sha,
        "raw_evidence_manifest_index_path": raw_path,
        "expected_raw_evidence_manifest_index_sha256": raw_sha,
        "collector_protocol_path": collector,
        "expected_collector_protocol_sha256": _sha256(collector),
        "statistical_protocol_path": statistical,
        "expected_statistical_protocol_sha256": _sha256(statistical),
        "requested_market_ids": list(manifest["ordered_market_ids"]),
        "repository_root": REPO_ROOT,
    }


def _moe_v2_r2_write_frozen_json(path: Path, payload: dict) -> Path:
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    path.with_suffix(".sha256").write_text(
        _sha256(path) + "\n",
        encoding="utf-8",
    )
    return path


def _moe_v2_runtime_prediction(required_path: str) -> dict:
    report = verify_frozen_json(MOE_V2_CONFIG_DIR / "moe_artifact_runtime_validation_report.json")
    return next(
        row
        for row in report["checks"]["prediction_results"]
        if row["required_path"] == required_path
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _moe_v2_r1_attempts(count: int) -> list[dict]:
    return [
        {
            "attempt_index": index,
            "market_id": f"market-{index:04d}",
            "market_start_ts": 1_800_000_000_000 + index * 900_000,
            "quality_valid": True,
            "raw_evidence_manifest_sha256": hashlib.sha256(
                f"evidence-{index}".encode()
            ).hexdigest(),
        }
        for index in range(1, count + 1)
    ]
