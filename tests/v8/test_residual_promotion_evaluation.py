from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from bigan.v8.polymarket.challenge_development_lane import sha256_file
from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.residual_promotion_evaluation import (
    _validate_evaluation_authorization,
    build_market_results,
    build_promotion_report,
    dry_run_evaluation_pipeline,
    validate_evaluation_execution_contract,
)
from examples.v8 import run_residual_promotion_evaluation as evaluation_cli

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = json.loads(
    (
        REPO_ROOT
        / "examples/v8/polymarket_configs"
        / "BTC-15M-cost-aware-market-residual-promotion-v1"
        / "prospective_statistical_protocol.json"
    ).read_text(encoding="utf-8")
)
CONFIG = (
    REPO_ROOT / "examples/v8/polymarket_configs" / "BTC-15M-cost-aware-market-residual-promotion-v1"
)


def _decision(market_id: str, *, accepted: bool, score: float = 0.05) -> dict:
    features = {
        "up_ask": 0.55,
        "up_bid": 0.53,
        "up_liquidity_depth": 10.0,
        "down_ask": 0.47,
        "down_bid": 0.45,
        "down_liquidity_depth": 10.0,
    }
    return {
        "market_id": market_id,
        "decision_ts": 2_000_000_000_000,
        "accepted": accepted,
        "selected_action": "BUY_UP_HOLD" if accepted else "NO_TRADE",
        "selected_side": "UP" if accepted else None,
        "selected_action_value": score if accepted else 0.0,
        "execution_features": features,
        "execution_features_sha256": canonical_json_sha256(features),
    }


def _population(count: int = 10) -> tuple[list[dict], list[dict], list[dict]]:
    candidate = []
    baseline = []
    settlements = []
    for index in range(count):
        market_id = f"market-{index:02d}"
        candidate.append(_decision(market_id, accepted=index % 2 == 0))
        baseline.append(_decision(market_id, accepted=False))
        settlements.append(
            {
                "market_id": market_id,
                "settlement_source": "official_polymarket",
                "official_resolution_reference": f"condition-{index:02d}",
                "settlement_finalized_at": "2030-01-01T00:00:00+00:00",
                "official_final": True,
                "inferred": False,
                "unresolved": False,
                "payout_up": 1.0 if index % 2 == 0 else 0.0,
                "payout_down": 0.0 if index % 2 == 0 else 1.0,
            }
        )
    return candidate, baseline, settlements


def test_market_pnl_uses_frozen_hold_to_settlement_cost_formula() -> None:
    candidate, baseline, settlements = _population()
    results, reconciliation = build_market_results(
        candidate_rows=candidate,
        baseline_rows=baseline,
        settlements=settlements,
        target_market_count=10,
    )
    expected = 1.0 - 0.55 - 0.0002 - 0.01 - 0.00005
    assert np.isclose(results[0]["candidate_unit_net_pnl"], expected)
    assert results[0]["baseline_unit_net_pnl"] == 0.0
    assert results[0]["paired_delta_unit_net_pnl"] == results[0]["candidate_unit_net_pnl"]
    assert reconciliation["passed"] is True
    assert reconciliation["paired_market_count"] == 10


def test_population_order_mismatch_fails_closed() -> None:
    candidate, baseline, settlements = _population()
    baseline[0], baseline[1] = baseline[1], baseline[0]
    with pytest.raises(ValueError, match="population identity mismatch"):
        build_market_results(
            candidate_rows=candidate,
            baseline_rows=baseline,
            settlements=settlements,
            target_market_count=10,
        )


def test_unresolved_or_inferred_settlement_fails_closed() -> None:
    candidate, baseline, settlements = _population()
    settlements[0]["unresolved"] = True
    with pytest.raises(ValueError, match="invalid or unresolved"):
        build_market_results(
            candidate_rows=candidate,
            baseline_rows=baseline,
            settlements=settlements,
            target_market_count=10,
        )
    settlements[0]["unresolved"] = False
    settlements[0]["inferred"] = True
    with pytest.raises(ValueError, match="invalid or unresolved"):
        build_market_results(
            candidate_rows=candidate,
            baseline_rows=baseline,
            settlements=settlements,
            target_market_count=10,
        )


def test_execution_feature_byte_drift_fails_closed() -> None:
    candidate, baseline, settlements = _population()
    candidate[0]["execution_features"]["up_ask"] = 0.56
    with pytest.raises(ValueError, match="feature SHA-256 mismatch"):
        build_market_results(
            candidate_rows=candidate,
            baseline_rows=baseline,
            settlements=settlements,
            target_market_count=10,
        )


def test_gate_failure_terminalizes_without_unlock() -> None:
    candidate, baseline, settlements = _population()
    settlements = copy.deepcopy(settlements)
    for row in settlements:
        row["payout_up"] = 0.0
        row["payout_down"] = 1.0
    results, reconciliation = build_market_results(
        candidate_rows=candidate,
        baseline_rows=baseline,
        settlements=settlements,
        target_market_count=10,
    )
    report = build_promotion_report(
        market_results=results,
        protocol=PROTOCOL,
        reconciliation=reconciliation,
        runtime_parity_passed=True,
        production=False,
        created_at="fixture",
    )
    assert report["all_gates_passed"] is False
    assert report["lineage_terminalized"] is True
    assert report["automatic_promotion_or_live_unlock"] is False
    assert report["micro_live_go_no_go"] == "NO_GO_LINEAGE_TERMINALIZED"
    assert report["promotion_evidence_eligible"] is False


def test_dry_run_is_deterministic_and_emits_no_gate_or_promotion_result() -> None:
    first = dry_run_evaluation_pipeline(protocol=PROTOCOL)
    second = dry_run_evaluation_pipeline(protocol=PROTOCOL)
    assert first == second
    assert first["population_alignment_passed"] is True
    assert first["five_blocks_exercised"] is True
    assert first["gate_results_emitted"] is False
    assert first["promotion_or_pass_result_emitted"] is False
    assert first["current_confirmatory_outcomes_accessed"] is False
    assert first["current_confirmatory_pnl_accessed"] is False
    assert first["automatic_promotion_or_live_unlock"] is False


def test_frozen_execution_contract_and_dry_run_artifacts_reconcile() -> None:
    for old_contract_path in (
        CONFIG / "promotion_evaluation_execution_contract.json",
        CONFIG / "promotion_evaluation_execution_contract_v2.json",
        CONFIG / "promotion_evaluation_execution_contract_v3.json",
    ):
        assert old_contract_path.with_suffix(".json.sha256").read_text().strip() == (
            sha256_file(old_contract_path)
        )
    contract_path = CONFIG / "promotion_evaluation_execution_contract_v4.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    validate_evaluation_execution_contract(contract, repository_root=REPO_ROOT)
    assert contract_path.with_suffix(".json.sha256").read_text().strip() == sha256_file(
        contract_path
    )
    report_path = CONFIG / "promotion_evaluation_dry_run_report.json"
    frozen_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert frozen_report == dry_run_evaluation_pipeline(protocol=PROTOCOL)
    assert report_path.with_suffix(".json.sha256").read_text().strip() == sha256_file(report_path)


def test_evaluation_entrypoint_is_bound_to_settlement_v4_contract() -> None:
    contract_path = evaluation_cli.EXECUTION_CONTRACT
    assert contract_path == CONFIG / "promotion_evaluation_execution_contract_v4.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert contract["contract_revision"] == "official_settlement_ingestion_v4"
    assert contract["finalization_correction"]["path"].endswith(
        "/finalization_native_missingness_correction.json"
    )
    assert contract["finalization_feature_envelope_correction"]["path"].endswith(
        "/finalization_feature_envelope_correction.json"
    )
    assert contract["settlement_ingestion_implementation"]["path"].endswith(
        "/residual_promotion_settlement.py"
    )
    assert contract_path.with_suffix(".json.sha256").read_text().strip() == (
        sha256_file(contract_path)
    )
    validate_evaluation_execution_contract(contract, repository_root=REPO_ROOT)


def test_evaluation_cli_passes_settlement_v4_contract_to_one_shot_runner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict = {}

    def _capture(**kwargs: object) -> dict:
        captured.update(kwargs)
        return {"evaluation_invoked": True}

    monkeypatch.setattr(evaluation_cli, "run_authorized_promotion_evaluation", _capture)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_residual_promotion_evaluation.py",
            "evaluate",
            "--service-root",
            str(tmp_path / "service"),
            "--freeze-dir",
            str(tmp_path / "freeze"),
            "--population-manifest-sha256",
            "1" * 64,
            "--settlement-ingestion-manifest",
            str(tmp_path / "settlement-manifest.json"),
            "--settlement-ingestion-manifest-sha256",
            "4" * 64,
            "--settlements",
            str(tmp_path / "settlements.jsonl"),
            "--settlements-sha256",
            "2" * 64,
            "--authorization",
            str(tmp_path / "authorization.json"),
            "--authorization-sha256",
            "3" * 64,
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )
    assert evaluation_cli.main() == 0
    assert captured["execution_contract_path"] == evaluation_cli.EXECUTION_CONTRACT
    assert captured["expected_execution_contract_sha256"] == sha256_file(
        evaluation_cli.EXECUTION_CONTRACT
    )


def test_execution_contract_implementation_drift_fails_closed() -> None:
    contract = json.loads(
        (CONFIG / "promotion_evaluation_execution_contract_v4.json").read_text(encoding="utf-8")
    )
    contract["implementation"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="descriptor SHA-256 mismatch"):
        validate_evaluation_execution_contract(contract, repository_root=REPO_ROOT)


def test_execution_contract_correction_child_drift_fails_closed() -> None:
    contract = json.loads(
        (CONFIG / "promotion_evaluation_execution_contract_v4.json").read_text(encoding="utf-8")
    )
    contract["finalization_feature_envelope_correction"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="descriptor SHA-256 mismatch"):
        validate_evaluation_execution_contract(contract, repository_root=REPO_ROOT)


def test_outcome_authorization_template_is_not_executable() -> None:
    for old_template in (
        CONFIG / "promotion_outcome_evaluation_authorization_template.json",
        CONFIG / "promotion_outcome_evaluation_authorization_template_v2.json",
        CONFIG / "promotion_outcome_evaluation_authorization_template_v3.json",
    ):
        assert old_template.with_suffix(".json.sha256").read_text().strip() == (
            sha256_file(old_template)
        )
    template = json.loads(
        (CONFIG / "promotion_outcome_evaluation_authorization_template_v4.json").read_text(
            encoding="utf-8"
        )
    )
    with pytest.raises(ValueError, match="invalid"):
        _validate_evaluation_authorization(
            template,
            execution_contract=template["execution_contract"],
            population_manifest_sha256="0" * 64,
        )


def test_finalization_native_missingness_correction_is_outcome_blind() -> None:
    path = CONFIG / "finalization_native_missingness_correction.json"
    correction = json.loads(path.read_text(encoding="utf-8"))
    assert path.with_suffix(".json.sha256").read_text().strip() == sha256_file(path)
    observation = correction["outcome_blind_observation"]
    assert observation["quality_valid"] is True
    assert observation["missing_feature_count"] == 12
    assert sum(observation["missing_feature_counts"].values()) == 12
    assert observation["missing_values_encoded_as_zero"] is False
    assert observation["outcomes_accessed"] is False
    assert observation["settlement_accessed"] is False
    assert observation["pnl_accessed"] is False
    assert correction["correction"] == {
        "collector_quality_eligibility_changed": False,
        "population_selection_changed": False,
        "model_prediction_behavior_changed": False,
        "statistical_gate_changed": False,
        "native_missing_values_remain_nan": True,
        "missing_values_encoded_as_zero": False,
        "finalizer_requires_nonnegative_integer_missing_counts": True,
        "finalizer_requires_missing_count_sum_reconciliation": True,
        "finalizer_requires_all_existing_quality_observations_true": True,
    }


def test_finalization_feature_envelope_correction_is_outcome_blind() -> None:
    path = CONFIG / "finalization_feature_envelope_correction.json"
    correction = json.loads(path.read_text(encoding="utf-8"))
    assert path.with_suffix(".json.sha256").read_text().strip() == sha256_file(path)
    observation = correction["outcome_blind_observation"]
    assert observation["quality_valid"] is True
    assert observation["feature_row_schema"]["execution_feature_envelope"] == ("features")
    assert observation["outcomes_accessed"] is False
    assert observation["settlement_accessed"] is False
    assert observation["pnl_accessed"] is False
    assert correction["correction"] == {
        "collector_quality_eligibility_changed": False,
        "population_selection_changed": False,
        "model_prediction_behavior_changed": False,
        "execution_values_changed": False,
        "statistical_gate_changed": False,
        "finalizer_reads_frozen_features_envelope": True,
        "required_execution_fields_unchanged": True,
        "missing_or_nonnumeric_execution_fields_fail_closed": True,
    }
