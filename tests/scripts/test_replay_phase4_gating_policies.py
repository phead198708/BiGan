from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "replay_phase4_gating_policies.py"

spec = importlib.util.spec_from_file_location("replay_phase4_gating_policies", SCRIPT)
assert spec is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)


def test_policy_replay_scores_signal_and_round_first_metrics() -> None:
    rows = [
        _row(
            event_id="blocked-cheap-opportunity",
            round_slug="round-1",
            decision_ts=1000,
            edge=0.60,
            token_probability=0.90,
            entry_ask=0.30,
            entry_worst_price=0.32,
            opportunity=True,
            gain=0.25,
        ),
        _row(
            event_id="allowed-no-opportunity",
            round_slug="round-1",
            decision_ts=2000,
            edge=0.70,
            token_probability=1.00,
            entry_ask=0.50,
            entry_worst_price=0.52,
            opportunity=False,
            gain=0.02,
        ),
        _row(
            event_id="allowed-opportunity",
            round_slug="round-2",
            decision_ts=3000,
            edge=0.50,
            token_probability=0.90,
            entry_ask=0.40,
            entry_worst_price=0.42,
            opportunity=True,
            settlement_opportunity=True,
            gain=0.20,
        ),
    ]

    current = module.PolicySpec(
        name="current",
        edge_threshold=0.45,
        min_entry_price=0.35,
    )
    relaxed_min = module.PolicySpec(
        name="relaxed_min",
        edge_threshold=0.45,
        min_entry_price=0.25,
    )

    current_metrics = module.score_policy(rows, current)
    relaxed_metrics = module.score_policy(rows, relaxed_min)

    assert current_metrics["candidate_count"] == 2
    assert current_metrics["opportunities_allowed"] == 1
    assert current_metrics["volatility_opportunities_allowed"] == 1
    assert current_metrics["settlement_opportunities_allowed"] == 1
    assert current_metrics["volatility_recall"] == 0.5
    assert current_metrics["settlement_recall"] == 1.0
    assert current_metrics["round_first_candidate_count"] == 2
    assert current_metrics["round_first_opportunities"] == 1

    assert relaxed_metrics["candidate_count"] == 3
    assert relaxed_metrics["opportunities_allowed"] == 2
    assert relaxed_metrics["volatility_opportunities_allowed"] == 2
    assert relaxed_metrics["settlement_opportunities_allowed"] == 1
    assert relaxed_metrics["round_first_candidate_count"] == 2
    assert relaxed_metrics["round_first_opportunities"] == 2
    assert relaxed_metrics["round_first_precision"] == 1.0


def test_cheap_token_policy_grid_scores_fresh_edge_gate() -> None:
    rows = [
        _row(
            event_id="strong-cheap",
            round_slug="round-1",
            decision_ts=1000,
            edge=0.01,
            token_probability=0.93,
            entry_ask=0.21,
            entry_worst_price=0.23,
            opportunity=True,
            gain=0.70,
        ),
        _row(
            event_id="weak-cheap",
            round_slug="round-2",
            decision_ts=2000,
            edge=0.01,
            token_probability=0.55,
            entry_ask=0.21,
            entry_worst_price=0.23,
            opportunity=True,
            gain=0.20,
        ),
    ]
    policies = {policy.name: policy for policy in module._cheap_token_policy_grid()}

    loose = module.score_policy(rows, policies["cheap_ask_0.20_fresh_0.40_seconds_420"])
    strict = module.score_policy(rows, policies["cheap_ask_0.20_fresh_0.50_seconds_420"])

    assert loose["candidate_count"] == 1
    assert strict["candidate_count"] == 1
    assert strict["opportunities_allowed"] == 1


def _row(
    *,
    event_id: str,
    round_slug: str,
    decision_ts: int,
    edge: float,
    token_probability: float,
    entry_ask: float,
    entry_worst_price: float,
    opportunity: bool,
    gain: float,
    settlement_opportunity: bool = False,
) -> dict:
    return {
        "event_id": event_id,
        "round_slug": round_slug,
        "decision_ts": decision_ts,
        "edge": edge,
        "token_probability": token_probability,
        "entry_ask": entry_ask,
        "entry_worst_price": entry_worst_price,
        "seconds_to_expiry_at_decision": 600.0,
        "outcome_side": "UP",
        "volatility_exit_opportunity": opportunity,
        "settlement_hold_opportunity": settlement_opportunity,
        "max_exit_gain": gain,
    }
