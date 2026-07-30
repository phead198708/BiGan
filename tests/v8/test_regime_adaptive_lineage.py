from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from bigan.v8.polymarket.moe_confirmatory_lineage import (
    assert_metric_payload_matches,
    deterministic_moe_route,
    frozen_expert_or_fallback,
)
from bigan.v8.polymarket.moe_static_artifact import (
    load_and_verify_static_moe_artifact,
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
CONFIG_DIR = (
    REPO_ROOT
    / "examples"
    / "v8"
    / "polymarket_configs"
    / "BTC-15M-regime-adaptive-v1"
)
EVALUATION_DIR = (
    REPO_ROOT
    / "examples"
    / "v8"
    / "polymarket_training_artifacts"
    / "BTC-15M-regime-adaptive-v1-development-evaluation"
)
MOE_CONFIG_DIR = (
    REPO_ROOT
    / "examples"
    / "v8"
    / "polymarket_configs"
    / "BTC-15M-MoE-confirmatory-v1"
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
    evaluation = verify_frozen_json(
        CONFIG_DIR / "rolling_origin_evaluation_protocol.json"
    )
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
    assert mixture["mean_unit_net_pnl_bootstrap_95pct_lower"] == pytest.approx(
        -0.03593672945205478
    )
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
    report = verify_frozen_json(
        MOE_CONFIG_DIR / "development_metric_reconciliation_report.json"
    )

    assert report["reconciliation_passed"] is True
    assert report["population"] == {
        "candidate_count": 5,
        "prediction_row_count": 1460,
        "fold_audit_count": 365,
        "oof_market_count": 73,
        "target_or_future_label_leakage_count": 0,
    }
    assert report["cost_reconciliation"]["row_decomposition_mismatch_count"] == 0
    assert all(
        item["comparison"]["passed"]
        for item in report["candidate_reconciliation"]
    )
    assert {
        item["candidate_id"] for item in report["candidate_reconciliation"]
    } == {
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
    assert report["parent_selection_reconciliation"][
        "fresh_collection_allowed"
    ] is False


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
    report = verify_frozen_json(
        MOE_CONFIG_DIR / "moe_route_attribution_report.json"
    )
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
    assert sum(row["unit_net_pnl"] for row in rows) == pytest.approx(
        report["total_unit_net_pnl"]
    )
    assert (
        report["pnl_attribution"]["native_expert_pnl"]
        + report["pnl_attribution"]["global_fallback_pnl"]
        == pytest.approx(report["total_unit_net_pnl"])
    )
    assert sum(row["selected_side"] == "UP" for row in accepted) + sum(
        row["selected_side"] == "DOWN" for row in accepted
    ) == len(accepted)
    assert (
        report["pnl_attribution"]["chronological_half_pnl"]["first"]
        + report["pnl_attribution"]["chronological_half_pnl"]["second"]
        == pytest.approx(report["total_unit_net_pnl"])
    )
    provider = report["provider_and_missingness"]
    assert (
        provider["accepted_complete_feature_market_count"]
        + provider["accepted_missing_feature_market_count"]
        == len(accepted)
    )
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
    assert candidate["hardening_inputs"]["metric_reconciliation"][
        "reconciliation_passed"
    ] is True
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
        "examples/v8/polymarket_configs/BTC-15M-MoE-confirmatory-v1/"
        "moe_artifact_graph.json"
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
        Path(descriptor["path"]).is_absolute()
        for descriptor in graph["artifacts"].values()
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
    graph = json.loads(
        (MOE_CONFIG_DIR / "moe_artifact_graph.json").read_text(encoding="utf-8")
    )
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
    graph = json.loads(
        (MOE_CONFIG_DIR / "moe_artifact_graph.json").read_text(encoding="utf-8")
    )
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

    assert template["artifact_role"] == (
        "inactive_template_only_not_collection_authority"
    )
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
    reporting = verify_frozen_json(
        MOE_CONFIG_DIR / "moe_future_evaluation_reporting_contract.json"
    )
    collector = verify_frozen_json(
        MOE_CONFIG_DIR / "moe_confirmatory_collector_protocol.json"
    )

    for descriptor in (
        protocol["frozen_inputs"]["router_contract"],
        protocol["frozen_inputs"]["feature_contract"],
        protocol["frozen_inputs"]["cost_contract"],
        protocol["frozen_inputs"]["collector_protocol"],
        protocol["frozen_inputs"]["future_evaluation_reporting_contract"],
    ):
        path = REPO_ROOT / descriptor["path"]
        assert path.is_file()
        assert _sha256(path) == descriptor["sha256"]
    candidate = protocol["frozen_inputs"]["candidate_artifact"]
    baseline = protocol["frozen_inputs"]["baseline_artifact"]
    assert candidate["bundle_hash"] == graph["bundle_hash"]
    assert candidate["artifact_graph_sha256"] == _sha256(
        MOE_CONFIG_DIR / "moe_artifact_graph.json"
    )
    assert baseline["sha256"] == graph["artifacts"]["moe_global_fallback.json"][
        "sha256"
    ]
    assert protocol["bootstrap"] == {
        "method": (
            "market_level_paired_percentile_bootstrap_with_shared_resample_indices"
        ),
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
    assert collector["quality_validity"][
        "paired_up_down_executable_ask_availability_is_not_a_selection_filter"
    ] is True
    assert protocol["safety"] == reporting["safety"] == collector["safety"] == SAFETY


def test_moe_future_reporting_contract_requires_all_unfiltered_panels() -> None:
    contract = verify_frozen_json(
        MOE_CONFIG_DIR / "moe_future_evaluation_reporting_contract.json"
    )
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
    assert contract["safety"] == SAFETY


def test_moe_authorization_template_pins_scope_but_grants_zero_authority() -> None:
    template = verify_frozen_json(
        MOE_CONFIG_DIR / "moe_fresh_collection_authorization_template.json"
    )
    graph = verify_frozen_json(MOE_CONFIG_DIR / "moe_artifact_graph.json")
    frozen = template["frozen_inputs"]

    assert frozen["artifact_bundle"]["bundle_hash"] == graph["bundle_hash"]
    assert frozen["router_contract"]["sha256"] == graph["artifacts"][
        "moe_router_contract.json"
    ]["sha256"]
    for route in ("high_vol", "bullish", "bearish", "low_vol"):
        assert frozen["expert_artifacts"][route]["sha256"] == graph["artifacts"][
            f"moe_expert_{route}.json"
        ]["sha256"]
    assert frozen["global_fallback"]["sha256"] == graph["artifacts"][
        "moe_global_fallback.json"
    ]["sha256"]
    assert frozen["feature_contract"]["sha256"] == graph["artifacts"][
        "moe_feature_contract.json"
    ]["sha256"]
    for name in ("statistical_protocol", "collector_protocol", "reporting_contract"):
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
    assert template["state"] == {
        "fresh_collection_authorized": False,
        "fresh_collection_started": False,
        "fresh_outcomes_opened": False,
    }
    assert template["safety"] == SAFETY


def test_moe_hardening_resume_preserves_provenance_limit_and_safety() -> None:
    resume = verify_frozen_json(
        MOE_CONFIG_DIR / "lineage_hardening_resume_record.json"
    )
    report = verify_frozen_json(MOE_CONFIG_DIR / "moe_hardening_report.json")

    assert resume["previous_blocked_record"]["preserved_as_historical_record"] is True
    assert resume["resume_basis"]["recorded_parent_source_commit_now_claimed_reachable"] is (
        False
    )
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
