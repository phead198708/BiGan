from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.residual_promotion_evaluation import (
    build_market_results,
    build_promotion_report,
    dry_run_evaluation_pipeline,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = json.loads(
    (
        REPO_ROOT
        / "examples/v8/polymarket_configs"
        / "BTC-15M-cost-aware-market-residual-promotion-v1"
        / "prospective_statistical_protocol.json"
    ).read_text(encoding="utf-8")
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
    assert results[0]["paired_delta_unit_net_pnl"] == results[0][
        "candidate_unit_net_pnl"
    ]
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
